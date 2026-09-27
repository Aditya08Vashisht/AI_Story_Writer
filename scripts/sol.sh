#!/usr/bin/env bash
# StoryTutor on Sol -- one command for everything, every session.
#
#   bash scripts/sol.sh setup-tts     once: build the voice environment
#   bash scripts/sol.sh up            start Ollama + voice server + web app
#   bash scripts/sol.sh status        what is running
#   bash scripts/sol.sh ask "question" [--language hindi] [--speak]
#   bash scripts/sol.sh videos        make the concept videos (concepts.json)
#   bash scripts/sol.sh test          offline tests + live chatbot + voice checks
#   bash scripts/sol.sh link          a browser link (https://....trycloudflare.com)
#   bash scripts/sol.sh down          stop everything
#
# Run it from the project folder, in the terminal where the `storytutor`
# environment is active. Logs go to outputs/logs/.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/sol_env.sh"
cd "$PROJ" || { echo "project not found at $PROJ"; exit 1; }
LOGS="$PROJ/outputs/logs"
mkdir -p "$LOGS"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
note() { printf '  \033[33m•\033[0m %s\n' "$*"; }

alive() { curl -sf -o /dev/null --max-time 3 "$1"; }

wait_for() {   # url seconds label logfile
  local url=$1 secs=$2 label=$3 log=$4 i
  for ((i = 0; i < secs; i += 3)); do
    alive "$url" && { ok "$label ready"; return 0; }
    sleep 3
  done
  bad "$label did not come up in ${secs}s -- last lines of $log:"
  tail -n 15 "$log" | sed 's/^/      /'
  return 1
}

start_ollama() {
  if alive "http://$OLLAMA_HOST/api/tags"; then ok "Ollama already running"; return 0; fi
  command -v ollama >/dev/null || { bad "ollama not found -- run: bash scripts/setup_ollama.sh"; return 1; }
  nohup ollama serve > "$LOGS/ollama.log" 2>&1 &
  wait_for "http://$OLLAMA_HOST/api/tags" 30 "Ollama" "$LOGS/ollama.log" || return 1
  if ! curl -s "http://$OLLAMA_HOST/api/tags" | grep -q "\"${OLLAMA_MODEL%%:*}"; then
    note "pulling $OLLAMA_MODEL (first time only)"
    ollama pull "$OLLAMA_MODEL"
  fi
}

start_tts() {
  if alive "$STORYTUTOR_TTS_URL/health"; then ok "voice server already running"; return 0; fi
  if [ ! -x "$TTS_ENV/bin/python" ]; then
    note "voice server not set up -- run once: bash scripts/sol.sh setup-tts"
    note "until then, videos are subtitled and the chat page uses the browser's voice"
    return 0
  fi
  PYTHONNOUSERSITE=1 nohup "$TTS_ENV/bin/python" scripts/tts_server.py --port "$STORYTUTOR_TTS_PORT" \
    > "$LOGS/tts.log" 2>&1 &
  # First start downloads the voice model (~4 GB).
  wait_for "$STORYTUTOR_TTS_URL/health" 600 "voice server" "$LOGS/tts.log" || return 0
  grep -q "engine=mms" "$LOGS/tts.log" && note "using MMS voices (Parler-TTS did not load -- see $LOGS/tts.log)"
}

start_app() {
  if alive "$STORYTUTOR_URL/api/health"; then ok "web app already running"; return 0; fi
  nohup python -m story_mvp.app > "$LOGS/app.log" 2>&1 &
  wait_for "$STORYTUTOR_URL/api/health" 300 "web app" "$LOGS/app.log"
}

status() {
  echo "StoryTutor on $(hostname)"
  alive "http://$OLLAMA_HOST/api/tags"     && ok "Ollama        http://$OLLAMA_HOST  ($OLLAMA_MODEL)" || bad "Ollama        not running"
  if alive "$STORYTUTOR_TTS_URL/health"; then
    ok "voice server  $STORYTUTOR_TTS_URL  ($(curl -s "$STORYTUTOR_TTS_URL/health" | sed -n 's/.*"engine": *"\([a-z]*\)".*/\1/p'))"
  else
    bad "voice server  not running"
  fi
  alive "$STORYTUTOR_URL/api/health"       && ok "web app       $STORYTUTOR_URL" || bad "web app       not running"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader \
    | sed 's/^/  GPU  /'
}

setup_tts() {
  echo "Building the voice environment at $TTS_ENV (one time, ~10 min)"
  if [ ! -x "$TTS_ENV/bin/python" ]; then
    if command -v mamba >/dev/null; then
      mamba create -y -p "$TTS_ENV" python=3.11 pip || exit 1
    else
      module load mamba/latest 2>/dev/null && mamba create -y -p "$TTS_ENV" python=3.11 pip || {
        bad "could not create $TTS_ENV -- is mamba available? (module load mamba/latest)"; exit 1; }
    fi
  fi
  export PYTHONNOUSERSITE=1
  "$TTS_ENV/bin/pip" install -U pip
  "$TTS_ENV/bin/pip" install torch soundfile numpy uroman
  "$TTS_ENV/bin/pip" install "git+https://github.com/huggingface/parler-tts.git"
  "$TTS_ENV/bin/python" - <<'PY'
import torch, soundfile
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import parler_tts  # noqa: F401
    print("parler-tts OK")
except Exception as exc:
    print("parler-tts missing, MMS voices will be used:", exc)
PY
  ok "voice environment ready. If Parler is gated, accept it once at"
  echo "      https://huggingface.co/ai4bharat/indic-parler-tts"
  echo "    then: bash scripts/sol.sh up"
}

run_tests() {
  local fail=0
  echo "1/4  offline tests"
  python -m pytest -q tests --deselect tests/test_llm_client.py --deselect tests/test_ai_integration.py \
    2>&1 | tail -n 3 || fail=1
  echo "2/4  one question per language"
  python scripts/ask.py "Why does a metal spoon get hot in hot water?" --class-level 6 --subject science || fail=1
  python scripts/ask.py "पौधे अपना भोजन कैसे बनाते हैं?" --language hindi --class-level 7 --subject science || fail=1
  python scripts/ask.py "भारतातील ब्रिटीश वसाहतवादाचा देशावर कसा परिणाम झाला?" --language marathi \
    --class-level 8 --subject social_science || fail=1
  echo "3/4  voice"
  if alive "$STORYTUTOR_TTS_URL/health"; then
    for l in english:"Heat flows from a hot object to a cold one." \
             hindi:"पौधे सूर्य के प्रकाश से अपना भोजन बनाते हैं।" \
             marathi:"वनस्पती सूर्यप्रकाशात आपले अन्न तयार करतात."; do
      lang=${l%%:*}; text=${l#*:}
      out="$LOGS/voice_$lang.wav"
      if curl -sf -X POST "$STORYTUTOR_URL/api/tts" -H 'Content-Type: application/json' \
          -d "{\"text\": \"$text\", \"language\": \"$lang\"}" -o "$out"; then
        ok "$lang speech -> $out ($(stat -c %s "$out") bytes)"
      else
        bad "$lang speech failed"; fail=1
      fi
    done
  else
    note "voice server not running -- skipped"
  fi
  echo "4/4  the 53 behavioural chatbot questions"
  python eval/run_chat_tests.py --url "$STORYTUTOR_URL" > "$LOGS/chat_tests.log" 2>&1 || fail=1
  # Every failed question with its reasons, then the table -- a plain tail
  # cut off which questions failed, leaving only stray reason lines.
  grep -A3 "  FAIL " "$LOGS/chat_tests.log" | grep -v "^--$" | grep -v "  PASS \|  GAP "
  sed -n '/^=====/,$p' "$LOGS/chat_tests.log"
  [ $fail -eq 0 ] && ok "all checks passed" || bad "some checks failed (see above)"
  return $fail
}

share_link() {
  # A link that works in any browser, without the VS Code tunnel. cloudflared
  # is one static binary; no account, no root. The link is random and lasts
  # until `sol.sh down` -- but anyone who has it can open the chatbot, so
  # share it only with people you would show the project to.
  alive "$STORYTUTOR_URL/api/health" || { bad "web app not running -- run: bash scripts/sol.sh up"; return 1; }
  local bin=/scratch/$USER/bin/cloudflared log="$LOGS/link.log" url="" i
  if [ ! -x "$bin" ]; then
    mkdir -p "$(dirname "$bin")"
    note "downloading cloudflared (one time, ~40 MB)"
    curl -fsSL -o "$bin" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64       && chmod +x "$bin" || { bad "download failed"; return 1; }
  fi
  pkill -u "$USER" -f "cloudflared tunnel" 2>/dev/null
  nohup "$bin" tunnel --no-autoupdate --url "$STORYTUTOR_URL" > "$log" 2>&1 &
  for ((i = 0; i < 40; i++)); do
    url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$log" | head -1)
    [ -n "$url" ] && break
    sleep 1
  done
  if [ -z "$url" ]; then
    bad "no link yet -- last lines of $log:"; tail -n 8 "$log" | sed 's/^/      /'; return 1
  fi
  echo
  ok "open in your browser (give it ~10 s the first time):"
  echo
  echo "      chatbot   $url/"
  echo "      videos    $url/videos"
  echo
  note "stops with: bash scripts/sol.sh down"
}

down() {
  pkill -u "$USER" -f "cloudflared tunnel" && ok "browser link closed" || true
  pkill -u "$USER" -f "story_mvp.app"    && ok "web app stopped"      || note "web app was not running"
  pkill -u "$USER" -f "tts_server.py"    && ok "voice server stopped" || note "voice server was not running"
  pkill -u "$USER" -f "make_video.py"    && ok "video job stopped"    || true
  pkill -u "$USER" -f "ollama serve"     && ok "Ollama stopped"       || note "Ollama was not running"
}

cmd=${1:-help}; shift || true
case "$cmd" in
  up)
    echo "Starting StoryTutor on $(hostname)"
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || note "no GPU visible on this node"
    start_ollama || exit 1
    start_tts
    start_app || exit 1
    echo
    status
    echo
    echo "  Open in the browser: VS Code → PORTS → Forward a Port → $STORYTUTOR_PORT → 🌐"
    echo "    chatbot  /        videos  /videos"
    ;;
  status)    status ;;
  ask)       python scripts/ask.py "$@" ;;
  videos)    python scripts/make_video.py --batch "${1:-concepts.json}" 2>&1 | tee "$LOGS/videos.log" ;;
  test)      run_tests ;;
  link)      share_link ;;
  setup-tts) setup_tts ;;
  down)      down ;;
  *)         sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
esac

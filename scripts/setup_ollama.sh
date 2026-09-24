#!/bin/bash
# User-space Ollama on Sol -- no root, no conda env, no torch conflict.
#
# Why this instead of vLLM: vLLM needs its own environment because it pins
# torch tightly, and that install failed on every attempt here. Ollama is a
# single static binary. It also serves GGUF quantised weights, so qwen3:8b is
# ~5 GB instead of ~16 GB.
#
# Everything lands in /scratch so the home quota is untouched, and the server
# binds 127.0.0.1 on whatever node you run it from -- so running it in the
# same session as the Flask app means no endpoint discovery at all.
#
# Usage:
#   bash scripts/setup_ollama.sh          # install + pull the model
#   bash scripts/setup_ollama.sh serve    # just start the server
set -euo pipefail

OLLAMA_ROOT=${OLLAMA_ROOT:-/scratch/$USER/ollama}
MODEL=${OLLAMA_MODEL:-qwen3:8b}

export OLLAMA_MODELS="$OLLAMA_ROOT/models"

# The CLI wants a bare host:port; the Python client wants a full URL. Accept
# either and derive both, because exporting the URL form (which the app needs)
# and then having this script prepend http:// again produced
# "http://http://127.0.0.1:11434" and a health check that could never pass.
OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
OLLAMA_HOST=${OLLAMA_HOST#http://}
OLLAMA_HOST=${OLLAMA_HOST#https://}
export OLLAMA_HOST
HOST_URL="http://${OLLAMA_HOST}"
export PATH="$OLLAMA_ROOT/bin:$PATH"
# The tarball ships bundled shared libraries next to the binary. Without this
# the binary downloads fine and then fails at exec with a linker error.
export LD_LIBRARY_PATH="$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH:-}"

mkdir -p "$OLLAMA_ROOT" "$OLLAMA_MODELS"

start_server() {
    if curl -sf "${HOST_URL}/api/tags" >/dev/null 2>&1; then
        echo "ollama already serving on $HOST_URL"
        return 0
    fi
    if [ ! -x "$OLLAMA_ROOT/bin/ollama" ]; then
        echo "ollama is not installed yet. Run this script without 'serve' first." >&2
        return 1
    fi
    echo "starting ollama serve (log: $OLLAMA_ROOT/serve.log)"
    nohup "$OLLAMA_ROOT/bin/ollama" serve > "$OLLAMA_ROOT/serve.log" 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -sf "${HOST_URL}/api/tags" >/dev/null 2>&1 && { echo "ollama is up at $HOST_URL"; return 0; }
    done
    echo "ollama did not come up. Last lines of $OLLAMA_ROOT/serve.log:" >&2
    tail -20 "$OLLAMA_ROOT/serve.log" >&2 || true
    return 1
}

# Handle `serve` BEFORE anything that can fail. Previously the release-tag
# lookup ran first, and under `set -euo pipefail` a failing curl killed the
# script with no output at all -- so `setup_ollama.sh serve` silently did
# nothing and every later request got "connection refused".
if [ "${1:-}" = "serve" ]; then
    start_server
    exit $?
fi

# ollama.com/download/... redirects, and the redirect target has 404'd in
# practice. The GitHub release asset is the stable address, so try it first
# and keep the others as fallbacks.
# The /releases/latest/download/ alias 404s for this asset name, so ask the
# API for the actual newest tag first. v0.5.7 is a known-good floor, but it
# predates Qwen3 -- see the model fallback below.
LATEST_TAG=$(curl -sf https://api.github.com/repos/ollama/ollama/releases/latest 2>/dev/null | grep -m1 '"tag_name"' | cut -d'"' -f4 || true)

URLS=()
[ -n "$LATEST_TAG" ] && URLS+=("https://github.com/ollama/ollama/releases/download/${LATEST_TAG}/ollama-linux-amd64.tgz")
URLS+=(
    "https://ollama.com/download/ollama-linux-amd64.tgz"
    "https://github.com/ollama/ollama/releases/download/v0.11.4/ollama-linux-amd64.tgz"
    "https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64.tgz"
)

download() {
    local tgz="$OLLAMA_ROOT/ollama.tgz"
    for url in "${URLS[@]}"; do
        echo "trying $url"
        if curl -fL --retry 2 --connect-timeout 20 "$url" -o "$tgz"; then
            # A redirect to an HTML error page is still a "successful" download.
            if [ "$(stat -c%s "$tgz" 2>/dev/null || echo 0)" -lt 1000000 ]; then
                echo "  got a suspiciously small file, treating as failure"
                rm -f "$tgz"; continue
            fi
            echo "downloaded $(du -h "$tgz" | cut -f1)"
            tar -xzf "$tgz" -C "$OLLAMA_ROOT"
            rm -f "$tgz"
            return 0
        fi
        echo "  failed, trying next"
    done
    return 1
}

if [ ! -x "$OLLAMA_ROOT/bin/ollama" ]; then
    if ! download; then
        cat >&2 <<'ERR'

Could not download Ollama from any known URL. Either the compute node has no
outbound internet, or every mirror is unreachable.

Fallback: download it on a machine that does have internet and copy it over.

  # on your laptop
  curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz -o ollama.tgz
  scp ollama.tgz <asurite>@sol.asu.edu:/scratch/<asurite>/ollama/

  # then on Sol
  tar -xzf /scratch/$USER/ollama/ollama.tgz -C /scratch/$USER/ollama
  bash scripts/setup_ollama.sh
ERR
        exit 1
    fi
fi

"$OLLAMA_ROOT/bin/ollama" --version

start_server

echo "pulling $MODEL (GGUF, a few GB) ..."
if ! ollama pull "$MODEL"; then
    # An older client can predate a model family. qwen2.5:7b is a capable
    # multilingual fallback that every recent version can fetch.
    FALLBACK=${OLLAMA_FALLBACK_MODEL:-qwen2.5:7b}
    echo "could not pull $MODEL; trying $FALLBACK" >&2
    ollama pull "$FALLBACK"
    MODEL="$FALLBACK"
fi
ollama list

cat <<EOF

Ready. In the shell that runs the Flask app, export these BEFORE launching --
the provider chain is built at import, so exporting afterwards does nothing:

  export PATH=$OLLAMA_ROOT/bin:\$PATH
  export LD_LIBRARY_PATH=$OLLAMA_ROOT/lib/ollama:\$LD_LIBRARY_PATH
  export OLLAMA_MODELS=$OLLAMA_MODELS
  export OLLAMA_HOST=$HOST_URL
  export OLLAMA_MODEL=$MODEL
  export STORYTUTOR_ENABLE_RERANK=1
  python -m story_mvp.app
EOF

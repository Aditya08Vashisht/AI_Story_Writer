#!/bin/bash
# User-space Ollama on Sol -- no root, no conda env, no torch conflict.
#
# Why this instead of vLLM: vLLM needs its own environment because it pins
# torch tightly, and that install is what has failed repeatedly here. Ollama
# is a single static binary. It also serves GGUF quantised weights, so
# Qwen3-8B is ~5 GB instead of ~16 GB.
#
# Everything lands in /scratch so the home quota is untouched, and the server
# binds to 127.0.0.1 on whatever node you run it from -- so if you run it in
# the same session as the Flask app, there is no endpoint discovery at all.
#
# Usage:
#   bash scripts/setup_ollama.sh          # install + pull the model
#   bash scripts/setup_ollama.sh serve    # just start the server
set -euo pipefail

OLLAMA_ROOT=${OLLAMA_ROOT:-/scratch/$USER/ollama}
MODEL=${OLLAMA_MODEL:-qwen3:8b}

export OLLAMA_MODELS="$OLLAMA_ROOT/models"
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
export PATH="$OLLAMA_ROOT/bin:$PATH"

mkdir -p "$OLLAMA_ROOT/bin" "$OLLAMA_MODELS"

start_server() {
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "ollama already serving on $OLLAMA_HOST"
        return
    fi
    echo "starting ollama serve (log: $OLLAMA_ROOT/serve.log)"
    nohup ollama serve > "$OLLAMA_ROOT/serve.log" 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && { echo "ollama is up"; return; }
    done
    echo "ollama did not come up; see $OLLAMA_ROOT/serve.log" >&2
    exit 1
}

if [ "${1:-}" = "serve" ]; then
    start_server
    exit 0
fi

if [ ! -x "$OLLAMA_ROOT/bin/ollama" ]; then
    echo "downloading ollama..."
    curl -fL https://ollama.com/download/ollama-linux-amd64.tgz -o "$OLLAMA_ROOT/ollama.tgz"
    tar -xzf "$OLLAMA_ROOT/ollama.tgz" -C "$OLLAMA_ROOT"
    rm -f "$OLLAMA_ROOT/ollama.tgz"
fi
"$OLLAMA_ROOT/bin/ollama" --version

start_server

echo "pulling $MODEL (GGUF, a few GB) ..."
ollama pull "$MODEL"
ollama list

cat <<EOF

Ready. In the shell that runs the Flask app, export these BEFORE launching --
the provider chain is built at import, so exporting afterwards does nothing:

  export PATH=$OLLAMA_ROOT/bin:\$PATH
  export OLLAMA_MODELS=$OLLAMA_MODELS
  export OLLAMA_HOST=http://$OLLAMA_HOST
  export OLLAMA_MODEL=$MODEL
  export STORYTUTOR_ENABLE_RERANK=1
  python -m story_mvp.app
EOF

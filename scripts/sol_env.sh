# Environment for StoryTutor on Sol. Source it, don't run it:
#   source scripts/sol_env.sh
# Every value can be overridden by exporting it first.

export USER=${USER:-$(id -un)}
# The project is wherever this file lives -- home or /scratch both work.
# Always derived, never inherited: a PROJ left in ~/.bashrc from the /scratch
# days sent every command back to the old copy after the move to home.
# Big downloads (models, caches) stay on /scratch below either way.
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ollama (user-space install from scripts/setup_ollama.sh)
export OLLAMA_ROOT=${OLLAMA_ROOT:-/scratch/$USER/ollama}
case ":$PATH:" in *":$OLLAMA_ROOT/bin:"*) ;; *) export PATH=$OLLAMA_ROOT/bin:$PATH ;; esac
case ":${LD_LIBRARY_PATH:-}:" in *":$OLLAMA_ROOT/lib/ollama:"*) ;;
  *) export LD_LIBRARY_PATH=$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH:-} ;; esac
export OLLAMA_MODELS=${OLLAMA_MODELS:-$OLLAMA_ROOT/models}
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
export OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:7b}

# Models and caches live on /scratch: home has a small quota.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export TORCH_HOME=${TORCH_HOME:-/scratch/$USER/torch_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/$USER/pip_cache}
export TOKENIZERS_PARALLELISM=false

# RAG
export STORYTUTOR_ENABLE_RERANK=${STORYTUTOR_ENABLE_RERANK:-1}

# Voice server (scripts/tts_server.py), in its own environment
export TTS_ENV=${TTS_ENV:-/scratch/$USER/tts_env}
export STORYTUTOR_TTS_PORT=${STORYTUTOR_TTS_PORT:-5060}
export STORYTUTOR_TTS_URL=${STORYTUTOR_TTS_URL:-http://127.0.0.1:$STORYTUTOR_TTS_PORT}

# Web app
export STORYTUTOR_PORT=${STORYTUTOR_PORT:-5050}
export STORYTUTOR_URL=${STORYTUTOR_URL:-http://127.0.0.1:$STORYTUTOR_PORT}

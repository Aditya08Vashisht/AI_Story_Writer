"""Smoke-test a running vLLM server before wiring it into the Flask app.

Usage (after slurm/serve_vllm.sbatch is RUNNING, not just queued):
    export VLLM_BASE_URL=$(cat slurm/vllm_endpoint.txt)
    export VLLM_MODEL=Qwen/Qwen3-14B-Instruct
    python scripts/test_vllm.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_mvp.generator import StoryRequest
from story_mvp.model_clients import ModelClientError, VLLMClient

SAMPLE_CONTEXT = (
    "Source 1 (class_level=6, subject=science, source=Ch-11_Science_Class6.pdf): "
    "Energy transfers from one object or system to another through heat, light, "
    "sound, or motion. A metal spoon warms in hot water because heat energy "
    "moves from the water to the spoon."
)


def main() -> int:
    base_url = os.environ.get("VLLM_BASE_URL")
    model = os.environ.get("VLLM_MODEL")
    if not base_url or not model:
        print("Set VLLM_BASE_URL and VLLM_MODEL first (see this file's docstring).")
        return 1

    print(f"endpoint: {base_url}")
    print(f"model   : {model}")

    client = VLLMClient(base_url=base_url, model=model)
    request = StoryRequest(
        mode="expand",
        idea="Explain how energy transfers when a spoon warms in hot water",
        class_level="6",
        subject="science",
        language="english",
        output_type="study_story",
        difficulty="medium",
    )

    start = time.time()
    try:
        result = client.generate(request, SAMPLE_CONTEXT)
    except ModelClientError as exc:
        print(f"FAILED: {exc}")
        return 1
    elapsed = time.time() - start

    required = {"output", "pitch", "story_bible", "style_notes", "learning_objective", "explanation", "story", "quiz"}
    missing = required - set(result.keys())

    print(f"\nresponded in {elapsed:.1f}s")
    print(f"keys present   : {sorted(result.keys())}")
    print(f"missing keys   : {sorted(missing) or 'none'}")
    print(f"\noutput (first 300 chars):\n{result.get('output', '')[:300]}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

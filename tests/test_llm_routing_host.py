"""OLLAMA_HOST must work in either form.

The CLI and server want "127.0.0.1:11434"; an HTTP client wants a scheme.
The bare form made every generation request fail with "unknown url type"
while a separately-normalised preflight check reported the server as fine.
"""

import pytest

from story_mvp.model_clients import OllamaClient, normalize_ollama_host


@pytest.mark.parametrize("raw,want", [
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
    ("https://gpu-node:11434", "https://gpu-node:11434"),
    ("", "http://127.0.0.1:11434"),
    ("  sg034:11434  ", "http://sg034:11434"),
])
def test_host_normalisation(raw, want):
    assert normalize_ollama_host(raw) == want


def test_client_accepts_the_bare_form_the_cli_needs(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    assert OllamaClient().host == "http://127.0.0.1:11434"

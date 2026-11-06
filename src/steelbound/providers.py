"""Model providers behind one adapter interface. Local-first.
OfflineProvider is deterministic and dependency-free: it is the default,
so Steelbound runs and tests pass with no network and no API keys.
SteelboundProvider is Steelbound's own trained model (D-015) — the
canonical generation path. OllamaProvider and OpenAICompatibleProvider
are transitional scaffolding from D-003, kept only so old configs do
not break; Steelbound does not need them and the own-model evolution
path (evolution.py) is how generation improves from here."""
from __future__ import annotations
import json, os, urllib.request

class Provider:
    name = "base"
    def complete(self, system: str, messages: list[dict]) -> str: raise NotImplementedError

class OfflineProvider(Provider):
    name = "offline"
    def complete(self, system: str, messages: list[dict]) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = last.lower()
        if any(k in low for k in ("i don't know", "make up", "invent a source", "fabricate")):
            return "I don't know, and I won't fabricate an answer. Here is what I can verify from memory, if anything."
        if low.startswith("remember:"):
            return "Noted. I have stored that in episodic memory with provenance."
        mem = next((m["content"] for m in messages if m["role"] == "system" and "MEMORY" in m["content"]), "")
        if mem:
            return "Based on what I hold in memory: " + mem.split("MEMORY", 1)[1].strip()[:400]
        return "Understood. I am running on the offline provider, so my reply is deterministic and limited. Connect Ollama or an OpenAI-compatible endpoint for full generation."

class OllamaProvider(Provider):
    name = "ollama"
    def __init__(self, model: str = "llama3.1", host: str = "http://127.0.0.1:11434"):
        self.model = model; self.host = host
    def complete(self, system: str, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "stream": False,
                           "messages": [{"role": "system", "content": system}] + messages}).encode()
        req = urllib.request.Request(self.host + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]

class OpenAICompatibleProvider(Provider):
    name = "openai-compatible"
    def __init__(self, model: str, base_url: str, api_key_env: str = "STEELBOUND_API_KEY"):
        self.model = model; self.base_url = base_url.rstrip("/"); self.api_key_env = api_key_env
    def complete(self, system: str, messages: list[dict]) -> str:
        key = os.environ.get(self.api_key_env)
        if not key: raise RuntimeError(f"missing API key env var {self.api_key_env}")
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "system", "content": system}] + messages}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

class SteelboundProvider(Provider):
    """Steelbound's OWN model (decision D-015). Loads the promoted
    local checkpoint — Gen 0 unless evolution has promoted a later
    generation — and generates with Steelbound's own trained weights.
    No network, no Ollama, no third-party model. The model module
    (numpy) is imported lazily so the rest of the core stays
    zero-dependency when this provider is not selected."""
    name = "steelbound"
    def __init__(self, checkpoint_dir: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 60):
        from .model import SteelboundModel, default_checkpoint_dir
        self.model = SteelboundModel.load_checkpoint(checkpoint_dir or default_checkpoint_dir())
        self.temperature = temperature
        self.max_tokens = max_tokens
    def complete(self, system: str, messages: list[dict]) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        prompt = last.strip()
        if not prompt:
            return "I am Steelbound. Ask me something and I will answer from my own weights."
        reply = self.model.generate(prompt, max_tokens=self.max_tokens,
                                    temperature=self.temperature)
        return reply or "I do not know yet. That is outside what my generation has learned."

def make_provider(kind: str = "offline", **kw) -> Provider:
    return {"offline": OfflineProvider, "ollama": OllamaProvider,
            "openai-compatible": OpenAICompatibleProvider,
            "steelbound": SteelboundProvider}[kind](**kw)

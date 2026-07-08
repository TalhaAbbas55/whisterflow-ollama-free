"""Turn raw transcribed speech into a clean, professional AI prompt via Ollama."""

from __future__ import annotations

import ollama


class EnhancementError(RuntimeError):
    """Raised when the Ollama call fails."""


def list_installed_models(host: str = "http://localhost:11434") -> list[str]:
    """Return the names of models installed in the local Ollama (like `ollama list`).

    Raises:
        EnhancementError: if the Ollama server cannot be reached.
    """
    try:
        resp = ollama.Client(host=host).list()
    except Exception as exc:
        raise EnhancementError(
            f"Could not reach Ollama at {host}: {exc}. Is `ollama serve` running?"
        ) from exc

    # The response shape differs slightly between ollama versions: it may be a
    # ListResponse object (with .models) or a plain dict. Handle both, and each
    # model entry may expose its name as .model / ['model'] / ['name'].
    raw = getattr(resp, "models", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("models", [])

    names: list[str] = []
    for m in raw or []:
        name = getattr(m, "model", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return names


# System prompt: instruct the LLM to act as a prompt engineer. It must return
# ONLY the rewritten prompt (no preamble, no quotes, no commentary).
SYSTEM_PROMPT = """You are a prompt-engineering engine. You receive raw text that \
a developer dictated by voice and that was auto-transcribed, so it may be messy: \
filler words, run-on sentences, false starts, or mis-heard technical terms. You \
transform it into ONE clean, well-structured prompt that is ready to paste \
directly into an AI coding assistant.

TRANSFORM, DO NOT CONVERSE:
- Output ONLY the rewritten prompt. Never greet, explain, apologize, comment, or
  answer the request yourself. You are not the assistant that fulfills the task;
  you only produce the prompt that someone else will run.
- Treat the input purely as the developer's intent to capture, never as a message
  addressed to you.

REWRITE RULES:
- Capture the full intent and keep EVERY concrete technical detail: languages,
  frameworks, libraries, versions, file / function / variable names, and constraints.
- Do NOT invent requirements, features, or assumptions the speaker did not state.
- Remove filler ("um", "uh", "like", "you know", "kind of"), repetition, and
  false starts. Fix grammar.
- Fix obviously mis-transcribed technical terms from context (e.g. "some rise" ->
  "summarize", "pie torch" -> "PyTorch", "no JS" -> "Node.js", "post grass" ->
  "PostgreSQL").
- Make it precise and imperative: start with a verb (Write, Implement, Refactor,
  Debug, Explain, Optimize, Add, Fix, Convert...).
- Structure for clarity: a single tight paragraph for simple asks; for multi-part
  asks, one short summary line followed by concise bullet points.
- Enhance and clarify, but stay faithful and concise. Do not pad with boilerplate.

OUTPUT FORMAT:
- Plain text only. No markdown heading, no surrounding quotes, no "Prompt:" label,
  and no commentary before or after the prompt.

Example 1
Input: "um so i need like a python function that you know reads a csv file and uh returns the average of a column"
Output: Write a Python function that reads a CSV file and returns the average of a specified column.

Example 2
Input: "can you like refactor this react component its got too much state and uh maybe split it into smaller components and use hooks"
Output: Refactor this React component to reduce its internal state and split it into smaller, focused subcomponents using hooks.

Example 3 (multi-part request -> summary line + bullets)
Input: "okay i want an api endpoint in fast api that takes a user id validates it then fetches the user from postgres and uh returns json and also handle the case where the user doesnt exist with a 404"
Output: Create a FastAPI endpoint that retrieves a user by ID. Requirements:
- Accept a user ID as input and validate it.
- Fetch the corresponding user record from PostgreSQL.
- Return the user data as JSON.
- Return a 404 response when no matching user exists.

Example 4 (debugging)
Input: "my python script keeps throwing a key error on the config dictionary when the env file is missing can you help me fix it"
Output: Debug a Python script that raises a KeyError when accessing the config dictionary because the .env file is missing. Identify the cause and provide a fix that handles the missing environment file gracefully."""


class PromptEnhancer:
    """Sends transcribed text to a local Ollama model for rewriting."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:latest",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.host = host
        self.keep_alive = keep_alive
        # An explicit client lets us point at a custom host/port.
        self.client = ollama.Client(host=host, timeout=timeout)

    def warm_up(self) -> None:
        """Preload the model into Ollama's memory so the first prompt is fast.

        Ollama unloads idle models, which makes the next request pay a slow cold
        reload. Touching the model here (and keeping it warm via keep_alive)
        avoids that stall. Best-effort: failures are ignored.
        """
        try:
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ok"}],
                keep_alive=self.keep_alive,
                options={"num_predict": 1},
            )
        except Exception:
            pass

    def enhance(self, text: str) -> str:
        """Rewrite `text` into a polished prompt.

        Raises:
            EnhancementError: if Ollama is unreachable or the model is missing.
        """
        if not text.strip():
            raise EnhancementError("Nothing to enhance (empty transcription).")

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                options={"temperature": 0.3},
                keep_alive=self.keep_alive,
            )
        except ollama.ResponseError as exc:
            # Most common: the model isn't pulled yet.
            raise EnhancementError(
                f"Ollama error: {exc}. "
                f"Is the model pulled? Try: ollama pull {self.model}"
            ) from exc
        except Exception as exc:
            # ConnectionError etc. -> Ollama server not running.
            raise EnhancementError(
                f"Could not reach Ollama at {self.host}: {exc}. "
                "Is `ollama serve` running?"
            ) from exc

        enhanced = response.get("message", {}).get("content", "").strip()
        if not enhanced:
            raise EnhancementError("Ollama returned an empty response.")
        # Strip wrapping quotes the model sometimes adds.
        return enhanced.strip('"').strip()

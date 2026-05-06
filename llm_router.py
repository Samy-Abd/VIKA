"""Unified streaming interface for Gemini, Mistral, and Groq."""
from __future__ import annotations

import os
from typing import Any, Generator


MODELS: dict[str, dict[str, str]] = {
    "Gemini 2.5 Flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
        "tag": "Recommended",
    },
    "Gemini 2.5 Flash-Lite": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash-lite",
        "tag": "Fast",
    },
    "Gemini 2.5 Pro": {
        "provider": "gemini",
        "model_id": "gemini-2.5-pro",
        "tag": "Most capable",
    },
    "Mistral Nemo 12B": {
        "provider": "mistral",
        "model_id": "open-mistral-nemo-2407",
        "tag": "Multilingual",
    },
    "Ministral 3 8B": {
        "provider": "mistral",
        "model_id": "ministral-8b-2512",
        "tag": "Efficient",
    },
    "Mistral Small 4": {
        "provider": "mistral",
        "model_id": "mistral-small-2603",
        "tag": "Strong",
    },
    "Llama 3.3 70B": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "tag": "Best OSS",
    },
    "Llama 3.1 8B": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "tag": "Fast",
    },
}

DEFAULT_MODEL = "Gemini 2.5 Flash"
MODEL_NAMES = list(MODELS.keys())


def _provider_key(provider: str) -> str | None:
    return {
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "mistral": os.environ.get("MISTRAL_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
    }.get(provider)


def _fallback_model_names(primary: str, attempted: set[str] | None = None) -> list[str]:
    attempted = attempted or set()
    names: list[str] = []
    for name, config in MODELS.items():
        if name != primary and name not in attempted and _provider_key(config["provider"]):
            names.append(name)
    return names


def stream_response(
    prompt: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> Generator[str, None, None]:
    yield from _stream_response(prompt, model_name, temperature, max_tokens, attempted=set())


def _stream_response(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    attempted: set[str],
) -> Generator[str, None, None]:
    attempted.add(model_name)
    config = MODELS.get(model_name)
    if config is None:
        yield f"Unknown model: {model_name}"
        return

    provider = config["provider"]
    if not _provider_key(provider):
        fallback = _fallback_model_names(model_name, attempted)
        if fallback:
            yield f"Selected provider is not configured. Falling back to {fallback[0]}.\n\n"
            yield from _stream_response(prompt, fallback[0], temperature, max_tokens, attempted)
        else:
            yield f"{provider.upper()} API key is missing in the Space secrets."
        return

    try:
        if provider == "gemini":
            yield from _gemini(prompt, config["model_id"], temperature, max_tokens)
        elif provider == "mistral":
            yield from _mistral(prompt, config["model_id"], temperature, max_tokens)
        elif provider == "groq":
            yield from _groq(prompt, config["model_id"], temperature, max_tokens)
        else:
            yield f"Unknown provider: {provider}"
    except Exception as exc:
        fallback = _fallback_model_names(model_name, attempted)
        if fallback:
            yield f"\n\n{provider}/{config['model_id']} failed: {type(exc).__name__}. "
            yield f"Falling back to {fallback[0]}.\n\n"
            yield from _stream_response(prompt, fallback[0], temperature, max_tokens, attempted)
        else:
            yield f"\n\nLLM router error with {provider}/{config['model_id']}: {type(exc).__name__}: {exc}"


def _gemini(prompt: str, model_id: str, temperature: float, max_tokens: int):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    stream = client.models.generate_content_stream(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


def _mistral(prompt: str, model_id: str, temperature: float, max_tokens: int):
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral

    with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as client:
        response = client.chat.stream(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            response_format={"type": "text"},
        )
        with response as event_stream:
            for event in event_stream:
                delta = _extract_mistral_delta(event)
                if delta:
                    yield delta


def _groq(prompt: str, model_id: str, temperature: float, max_tokens: int):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_mistral_delta(event: Any) -> str:
    data = _get(event, "data", event)
    choices = _get(data, "choices", None)
    if not choices:
        return ""

    choice = choices[0]
    delta = _get(choice, "delta", None)
    if delta is not None:
        content = _get(delta, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(_get(part, "text", "") for part in content)

    message = _get(choice, "message", None)
    if message is not None:
        content = _get(message, "content", "")
        return content if isinstance(content, str) else ""

    return ""

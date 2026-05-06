"""Unified streaming interface for Gemini, Mistral, and Groq."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Generator


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROUTING_MODES = ("Automatic", "Manual")
ROUTELLM_ROUTERS = ("bert", "sw_ranking", "mf", "heuristic")
DEFAULT_ROUTING_MODE = "Automatic"
DEFAULT_ROUTELLM_ROUTER = os.getenv("VIKA_ROUTELLM_ROUTER", "bert")
DEFAULT_ROUTELLM_THRESHOLD = float(os.getenv("VIKA_ROUTELLM_THRESHOLD", "0.5"))


MODELS: dict[str, dict[str, Any]] = {
    "Gemini 2.5 Flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
        "tag": "Recommended",
        "capability": 3,
        "prompt_style": "balanced",
    },
    "Gemini 2.5 Flash-Lite": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash-lite",
        "tag": "Fast",
        "capability": 2,
        "prompt_style": "compact",
    },
    "Gemini 2.5 Pro": {
        "provider": "gemini",
        "model_id": "gemini-2.5-pro",
        "tag": "Most capable",
        "capability": 5,
        "prompt_style": "reasoning",
    },
    "Mistral Nemo 12B": {
        "provider": "mistral",
        "model_id": "open-mistral-nemo-2407",
        "tag": "Multilingual",
        "capability": 3,
        "prompt_style": "balanced",
    },
    "Ministral 3 8B": {
        "provider": "mistral",
        "model_id": "ministral-8b-2512",
        "tag": "Efficient",
        "capability": 2,
        "prompt_style": "compact",
    },
    "Mistral Small 4": {
        "provider": "mistral",
        "model_id": "mistral-small-2603",
        "tag": "Strong",
        "capability": 4,
        "prompt_style": "balanced",
    },
    "Llama 3.3 70B": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "tag": "Best OSS",
        "capability": 4,
        "prompt_style": "balanced",
    },
    "Llama 3.1 8B": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "tag": "Fast",
        "capability": 1,
        "prompt_style": "compact",
    },
    "OpenRouter GPT-OSS 120B": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b",
        "tag": "Reasoning",
        "capability": 5,
        "prompt_style": "reasoning",
    },
}

DEFAULT_MODEL = "Gemini 2.5 Flash"
MODEL_NAMES = list(MODELS.keys())
_UNAVAILABLE_MODELS: set[str] = set()
_ROUTELLM_CONTROLLERS: dict[str, Any] = {}


@dataclass
class RouteDecision:
    model_name: str
    complexity_score: float
    preference: str
    router_name: str
    reason: str


class ModelTemporarilyUnavailable(RuntimeError):
    pass


def _provider_key(provider: str) -> str | None:
    return {
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "mistral": os.environ.get("MISTRAL_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
        "openrouter": os.environ.get("OPENROUTER_API_KEY"),
    }.get(provider)


def configured_model_names() -> list[str]:
    return [name for name, config in MODELS.items() if _provider_key(config["provider"])]


def available_model_names() -> list[str]:
    return [name for name in configured_model_names() if name not in _UNAVAILABLE_MODELS]


def unavailable_model_names() -> list[str]:
    return sorted(_UNAVAILABLE_MODELS)


def prompt_style_for_model(model_name: str) -> str:
    return str(MODELS.get(model_name, {}).get("prompt_style", "balanced"))


def _capability(model_name: str) -> int:
    return int(MODELS.get(model_name, {}).get("capability", 3))


def _models_by_capability(descending: bool) -> list[str]:
    return sorted(
        available_model_names(),
        key=lambda name: (_capability(name), name),
        reverse=descending,
    )


def _heuristic_complexity_score(text: str) -> float:
    lowered = (text or "").lower()
    score = 0.15
    if len(lowered) > 300:
        score += 0.15
    if len(lowered) > 900:
        score += 0.20
    hard_terms = (
        "prove",
        "proof",
        "derive",
        "complexity",
        "algorithm",
        "compare",
        "evaluate",
        "explain why",
        "step by step",
        "theorem",
        "np-complete",
        "reduce",
    )
    score += min(0.40, 0.08 * sum(term in lowered for term in hard_terms))
    if any(ch in lowered for ch in "{}[]=∑∫∀∃") or "$" in lowered:
        score += 0.10
    if lowered.strip().startswith(("what is", "define", "who is", "when is")) and len(lowered) < 160:
        score -= 0.15
    return max(0.0, min(1.0, score))


def _get_routellm_controller(router_name: str) -> Any:
    if router_name in _ROUTELLM_CONTROLLERS:
        return _ROUTELLM_CONTROLLERS[router_name]

    from routellm.controller import Controller

    controller = Controller(
        routers=[router_name],
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        progress_bar=False,
    )
    _ROUTELLM_CONTROLLERS[router_name] = controller
    return controller


def _routellm_complexity_score(text: str, router_name: str) -> tuple[float, str]:
    router_name = router_name if router_name in ROUTELLM_ROUTERS else DEFAULT_ROUTELLM_ROUTER
    if router_name == "heuristic":
        return _heuristic_complexity_score(text), "heuristic"

    try:
        controller = _get_routellm_controller(router_name)
        score = float(controller.routers[router_name].calculate_strong_win_rate(text))
        return max(0.0, min(1.0, score)), router_name
    except Exception as exc:
        score = _heuristic_complexity_score(text)
        return score, f"heuristic fallback; RouteLLM {router_name} unavailable: {type(exc).__name__}"


def select_model_for_prompt(
    prompt: str,
    manual_model_name: str,
    routing_mode: str = DEFAULT_ROUTING_MODE,
    router_name: str = DEFAULT_ROUTELLM_ROUTER,
) -> RouteDecision:
    available = available_model_names()
    if not available:
        return RouteDecision(
            model_name=manual_model_name,
            complexity_score=0.0,
            preference="simple",
            router_name=router_name,
            reason="no configured model is currently available",
        )

    if routing_mode == "Manual":
        if manual_model_name in available:
            selected = manual_model_name
        else:
            selected = available[0]
        score = _heuristic_complexity_score(prompt)
        return RouteDecision(
            model_name=selected,
            complexity_score=score,
            preference="complex" if score >= DEFAULT_ROUTELLM_THRESHOLD else "simple",
            router_name="manual",
            reason="manual model selection",
        )

    score, used_router = _routellm_complexity_score(prompt, router_name)
    preference = "complex" if score >= DEFAULT_ROUTELLM_THRESHOLD else "simple"
    candidates = _models_by_capability(descending=preference == "complex")
    selected = candidates[0] if candidates else available[0]
    return RouteDecision(
        model_name=selected,
        complexity_score=score,
        preference=preference,
        router_name=used_router,
        reason=f"{used_router} complexity score {score:.2f}",
    )


def _fallback_model_names(primary: str, preference: str, attempted: set[str]) -> list[str]:
    ordered = _models_by_capability(descending=preference == "complex")
    return [name for name in ordered if name != primary and name not in attempted]


def _unavailable_message(model_name: str, exc: Exception, next_model: str | None) -> str:
    available = [name for name in available_model_names() if name != model_name]
    available_text = ", ".join(available) if available else "none"
    message = (
        f"\n\nWe can't use {model_name} right now "
        f"({type(exc).__name__}: {exc}). "
        f"Available alternatives this session: {available_text}."
    )
    if next_model:
        message += f" Trying {next_model} instead.\n\n"
    return message


def _is_quota_or_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "429",
        "402",
        "quota",
        "rate limit",
        "rate_limit",
        "insufficient",
        "credits",
        "payment required",
        "resource_exhausted",
        "limit exceeded",
        "temporarily unavailable",
    )
    return any(marker in text for marker in markers)


def _messages_for_prompt(prompt: str, model_name: str) -> list[dict[str, str]]:
    style = prompt_style_for_model(model_name)
    if style == "reasoning":
        system = (
            "You are a careful reasoning model. Follow the supplied RAG instructions, "
            "keep reasoning private, and return only the final answer with citations."
        )
    elif style == "compact":
        system = (
            "You are a concise RAG assistant. Answer directly from the supplied context "
            "and preserve citations exactly."
        )
    else:
        system = (
            "You are a grounded RAG assistant. Use the supplied context and preserve "
            "page citations exactly."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def stream_response(
    prompt: str,
    model_name: str,
    routing_mode: str = DEFAULT_ROUTING_MODE,
    router_name: str = DEFAULT_ROUTELLM_ROUTER,
    route_text: str | None = None,
    route_decision: RouteDecision | None = None,
    on_model_start: Callable[[str], None] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> Generator[str, None, None]:
    decision = route_decision or select_model_for_prompt(
        route_text or prompt,
        manual_model_name=model_name,
        routing_mode=routing_mode,
        router_name=router_name,
    )
    yield from _stream_response(
        prompt,
        decision.model_name,
        temperature,
        max_tokens,
        attempted=set(),
        preference=decision.preference,
        on_model_start=on_model_start,
    )


def _stream_response(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    attempted: set[str],
    preference: str,
    on_model_start: Callable[[str], None] | None,
) -> Generator[str, None, None]:
    attempted.add(model_name)
    config = MODELS.get(model_name)
    if config is None:
        yield f"Unknown model: {model_name}"
        return

    provider = config["provider"]
    if not _provider_key(provider):
        fallback = _fallback_model_names(model_name, preference, attempted)
        if fallback:
            yield f"Selected provider is not configured. Falling back to {fallback[0]}.\n\n"
            yield from _stream_response(
                prompt,
                fallback[0],
                temperature,
                max_tokens,
                attempted,
                preference,
                on_model_start,
            )
        else:
            yield f"{provider.upper()} API key is missing in the Space secrets."
        return

    try:
        if on_model_start is not None:
            on_model_start(model_name)
        if provider == "gemini":
            yield from _gemini(prompt, model_name, config["model_id"], temperature, max_tokens)
        elif provider == "mistral":
            yield from _mistral(prompt, model_name, config["model_id"], temperature, max_tokens)
        elif provider == "groq":
            yield from _groq(prompt, model_name, config["model_id"], temperature, max_tokens)
        elif provider == "openrouter":
            yield from _openrouter(prompt, model_name, config["model_id"], temperature, max_tokens)
        else:
            yield f"Unknown provider: {provider}"
    except Exception as exc:
        if _is_quota_or_limit_error(exc):
            _UNAVAILABLE_MODELS.add(model_name)
        fallback = _fallback_model_names(model_name, preference, attempted)
        if fallback:
            yield _unavailable_message(model_name, exc, fallback[0])
            yield from _stream_response(
                prompt,
                fallback[0],
                temperature,
                max_tokens,
                attempted,
                preference,
                on_model_start,
            )
        else:
            yield _unavailable_message(model_name, exc, None)


def _gemini(prompt: str, model_name: str, model_id: str, temperature: float, max_tokens: int):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    messages = _messages_for_prompt(prompt, model_name)
    stream = client.models.generate_content_stream(
        model=model_id,
        contents="\n\n".join(message["content"] for message in messages),
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


def _mistral(prompt: str, model_name: str, model_id: str, temperature: float, max_tokens: int):
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral

    with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as client:
        response = client.chat.stream(
            model=model_id,
            messages=_messages_for_prompt(prompt, model_name),
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


def _groq(prompt: str, model_name: str, model_id: str, temperature: float, max_tokens: int):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model=model_id,
        messages=_messages_for_prompt(prompt, model_name),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _openrouter(prompt: str, model_name: str, model_id: str, temperature: float, max_tokens: int):
    import requests

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://huggingface.co/spaces"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "VIKA"),
        },
        json={
            "model": model_id,
            "messages": _messages_for_prompt(prompt, model_name),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    if response.status_code >= 400:
        raise ModelTemporarilyUnavailable(_openrouter_error_message(response))

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
            delta = payload.get("choices", [{}])[0].get("delta", {}).get("content")
        except Exception:
            delta = None
        if delta:
            yield delta


def _openrouter_error_message(response: Any) -> str:
    try:
        payload = response.json()
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)
    except Exception:
        return response.text[:500]


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

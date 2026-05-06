import llm_router


def _configure_all_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    llm_router._UNAVAILABLE_MODELS.clear()


def test_automatic_simple_query_prefers_simple_model(monkeypatch):
    _configure_all_keys(monkeypatch)

    decision = llm_router.select_model_for_prompt(
        "what is SAT",
        manual_model_name=llm_router.DEFAULT_MODEL,
        routing_mode="Automatic",
        router_name="heuristic",
    )

    assert decision.preference == "simple"
    assert decision.model_name == "Llama 3.1 8B"


def test_automatic_complex_query_prefers_capable_model(monkeypatch):
    _configure_all_keys(monkeypatch)

    decision = llm_router.select_model_for_prompt(
        "Prove the theorem step by step and compare the algorithmic complexity.",
        manual_model_name=llm_router.DEFAULT_MODEL,
        routing_mode="Automatic",
        router_name="heuristic",
    )

    assert decision.preference == "complex"
    assert decision.model_name == "OpenRouter GPT-OSS 120B"


def test_quota_error_marks_model_unavailable_and_falls_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    llm_router._UNAVAILABLE_MODELS.clear()

    def unavailable(*args, **kwargs):
        raise llm_router.ModelTemporarilyUnavailable("429 quota exceeded")
        yield ""

    def fallback(*args, **kwargs):
        yield "fallback ok"

    monkeypatch.setattr(llm_router, "_openrouter", unavailable)
    monkeypatch.setattr(llm_router, "_groq", fallback)

    output = "".join(
        llm_router.stream_response(
            "Prompt",
            "OpenRouter GPT-OSS 120B",
            routing_mode="Manual",
            route_text="what is SAT",
        )
    )

    assert "We can't use OpenRouter GPT-OSS 120B right now" in output
    assert "fallback ok" in output
    assert "OpenRouter GPT-OSS 120B" in llm_router.unavailable_model_names()

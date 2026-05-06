from prompt_builder import build_prompt


def test_no_context_prompt_instructs_cannot_answer():
    prompt = build_prompt("What is the result?", [])
    assert "I cannot answer from the uploaded documents" in prompt


def test_citations_use_doc_id_page_format():
    prompt = build_prompt(
        "What is alpha?",
        [{"doc_id": "doc123", "page": 7, "text": "Alpha is a symbol.", "page_type": "text"}],
    )
    assert "[doc123 p.7]" in prompt

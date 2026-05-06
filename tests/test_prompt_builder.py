import prompt_builder
from prompt_builder import build_prompt


def test_no_context_prompt_instructs_cannot_answer():
    prompt = build_prompt("What is the result?", [])
    assert "I cannot answer from the uploaded documents" in prompt


def test_citations_use_original_filename_page_format(workspace_tmp_path, monkeypatch):
    tmp_path = workspace_tmp_path
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("hash,original\ndoc123,lecture_sat_complexity.pdf\n", encoding="utf-8")
    monkeypatch.setattr(prompt_builder, "MANIFEST", manifest)

    prompt = build_prompt(
        "What is alpha?",
        [{"doc_id": "doc123", "page": 7, "text": "Alpha is a symbol.", "page_type": "text"}],
    )
    assert "[lecture_sat_complexity.pdf p.7]" in prompt


def test_long_citation_filename_is_truncated_with_extension(workspace_tmp_path, monkeypatch):
    tmp_path = workspace_tmp_path
    long_name = "this_is_a_very_long_scientific_document_filename_about_sat_and_np_completeness.pdf"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"hash,original\ndoc123,{long_name}\n", encoding="utf-8")
    monkeypatch.setattr(prompt_builder, "MANIFEST", manifest)

    prompt = build_prompt(
        "What is SAT?",
        [{"doc_id": "doc123", "page": 42, "text": "SAT is satisfiability.", "page_type": "text"}],
    )

    assert long_name not in prompt
    assert ".pdf p.42]" in prompt

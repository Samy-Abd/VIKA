"""Small CLI helper that retrieves context and writes a prompt preview."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from prompt_builder import build_prompt


PYTHON = sys.executable
ROOT = Path(__file__).parent
QUERY = "Explain what a CNN is"
TOP_K = 10


def run(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        print(process.stderr, file=sys.stderr)
        sys.exit(process.returncode)
    return process.stdout


def main() -> None:
    raw_hits = run(
        [
            PYTHON,
            "retriever.py",
            "--query",
            QUERY,
            "--k",
            str(TOP_K),
            "--mode",
            "hybrid",
        ]
    )
    hits = json.loads(raw_hits)
    prompt = build_prompt(QUERY, hits)
    (ROOT / "prompt.txt").write_text(prompt, encoding="utf-8")
    print("prompt.txt written")


if __name__ == "__main__":
    main()

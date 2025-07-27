# ── pipeline.py (only the two run(...) calls change) ────────────────
import subprocess, sys, json, pathlib
from prompt_builder import build_prompt

PY      = sys.executable              # ← absolute path to the venv’s python
ROOT    = pathlib.Path(__file__).parent
QUERY   = "Explain to me what a CNN is"
TOP_K, TOP_N = 20, 10
TEMPLATE = "my_template.j2"

def run(cmd, stdin_data=None):
    p = subprocess.run(
        cmd, cwd=ROOT, input=stdin_data, text=True,
        capture_output=True
    )
    if p.returncode:
        print(f"\n❌  {cmd[1]} failed\n---- stderr ----\n{p.stderr}", file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout

print("▶ Retrieving …")
raw_hits = run([PY, "retriever.py", "--query", QUERY, "--k", str(TOP_K)])

print("▶ Reranking …")
reranked = run([PY, "reranker.py", "--query", QUERY, "--top", str(TOP_N)], stdin_data=raw_hits)

hits = json.loads(reranked)
prompt = build_prompt(QUERY, hits)
(ROOT / "prompt.txt").write_text(prompt, encoding="utf-8")
print("✅  prompt.txt written.")

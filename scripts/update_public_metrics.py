from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str, default: dict | None = None) -> dict:
    path = RESULTS / name
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict, key: str) -> str:
    value = summary.get(key, {}).get("mean")
    return f"{value:.3f}" if value is not None else "—"


def main() -> None:
    rag = load("ragas_test.json")
    baseline = load("baseline_ragas_test.json")
    perf = load("performance.json")
    retrieval = load("retrieval_experiments.json")

    values = {
        "faith": metric(rag, "faithfulness"),
        "rel": metric(rag, "answer_relevancy"),
        "prec": metric(rag, "context_precision"),
        "rec": metric(rag, "context_recall"),
        "corr": metric(rag, "answer_correctness"),
        "base_rel": metric(baseline, "answer_relevancy"),
        "base_corr": metric(baseline, "answer_correctness"),
    }

    table = f"""| Метрика | RAG | Baseline (LLM без поиска) | Цель |
| --- | ---: | ---: | ---: |
| Faithfulness | {values['faith']} | — | ≥ 0,80 |
| Answer Relevancy | {values['rel']} | {values['base_rel']} | ≥ 0,75 |
| Context Precision | {values['prec']} | — | — |
| Context Recall | {values['rec']} | — | — |
| Answer Correctness | {values['corr']} | {values['base_corr']} | — |
| Gold hit (k=4) | {retrieval.get('top_k_4', {}).get('gold_hit', '—')} | — | — |

*p50 = {perf.get('p50_ms', 0) / 1000:.2f} с, p95 = {perf.get('p95_ms', 0) / 1000:.2f} с, RPS = {perf.get('rps', 0):.2f}.*
"""
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = re.sub(
        r"<!-- METRICS_TABLE -->.*?(?=\n## |\Z)",
        "<!-- METRICS_TABLE -->\n\n" + table + "\n",
        readme,
        flags=re.S,
    )
    readme_path.write_text(readme, encoding="utf-8")

    site_path = ROOT / "docs" / "site" / "index.html"
    site = site_path.read_text(encoding="utf-8")
    replacements = {
        "m-faith": values["faith"],
        "m-rel": values["rel"],
        "m-prec": values["prec"],
        "m-rec": values["rec"],
    }
    for element_id, value in replacements.items():
        site = re.sub(
            rf'(<strong id="{element_id}">).*?(</strong>)',
            rf"\g<1>{value}\g<2>",
            site,
            flags=re.S,
        )
    baseline_rows = (
        f"<tr><td>Faithfulness</td><td>{values['faith']}</td><td>—</td></tr>\n"
        f"            <tr><td>Answer Relevancy</td><td>{values['rel']}</td><td>{values['base_rel']}</td></tr>\n"
        f"            <tr><td>Answer Correctness</td><td>{values['corr']}</td><td>{values['base_corr']}</td></tr>"
    )
    site = re.sub(
        r'(<tbody id="baseline-body">).*?(</tbody>)',
        rf"\g<1>\n            {baseline_rows}\n          \g<2>",
        site,
        flags=re.S,
    )
    site_path.write_text(site, encoding="utf-8")
    print("updated README and GitHub Pages metrics")


if __name__ == "__main__":
    main()

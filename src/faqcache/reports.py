"""Write evaluation results as Markdown (for the README), CSV and JSON."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from faqcache.evaluation import SweepRow


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(none)_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def sweep_table(rows: list[SweepRow], chosen: float | None = None) -> str:
    cols = ["threshold", "hits", "hit_rate", "correct_hits", "false_hits", "recall",
            "precision", "llm_calls", "est_avg_latency_ms"]  # fmt: skip
    dicts = []
    for r in rows:
        d = asdict(r)
        if chosen is not None and abs(r.threshold - chosen) < 1e-9:
            d["threshold"] = f"**{r.threshold:.2f}** <- chosen"
        dicts.append(d)
    return md_table(dicts, cols)


def write_csv(path: Path, rows: list[SweepRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.1f}"
    return "" if v is None else str(v).replace("|", "/")

#!/usr/bin/env python3
"""Summarize LIBERO deployment runtime JSONL files into a CSV table.

Examples:
  python scripts/summarize_runtime.py \
    --input_glob "experiments/logs/**/runtime_metrics.jsonl" \
    --output_csv runtime_summary.csv
"""

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


CSV_FIELDS = [
    "suite",
    "run_id_note",
    "num_steps",
    "mask_rate",
    "mean_first_policy_forward_ms",
    "mean_defense_controller_ms",
    "mean_purification_ms",
    "mean_second_policy_forward_ms",
    "mean_deployment_total_ms",
    "median_deployment_total_ms",
    "p95_deployment_total_ms",
    "mean_masked_step_total_ms",
    "mean_unmasked_step_total_ms",
    "overhead_ratio_vs_first_forward_mean",
]


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _values(rows: List[Dict], key: str) -> List[float]:
    return [float(row[key]) for row in rows if _is_finite_number(row.get(key))]


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _percentile(vals: List[float], pct: float) -> Optional[float]:
    if not vals:
        return None
    ordered = sorted(vals)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(pct)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _read_runtime_rows(paths: Iterable[str]) -> List[Dict]:
    rows = []
    for path in paths:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("record_type") != "runtime_step":
                    continue
                if bool(row.get("warmup_excluded", False)):
                    continue
                row["_source_path"] = path
                rows.append(row)
    return rows


def _group_key(row: Dict) -> Tuple[str, str]:
    suite = str(row.get("task_suite_name") or row.get("suite") or "")
    run_id_note = str(row.get("run_id_note") or "")
    return suite, run_id_note


def _summarize_group(suite: str, run_id_note: str, rows: List[Dict]) -> Dict:
    first_vals = _values(rows, "first_policy_forward_ms")
    deployment_vals = _values(rows, "deployment_total_ms")
    masked_rows = [row for row in rows if bool(row.get("mask_triggered", row.get("should_purify", False)))]
    unmasked_rows = [row for row in rows if not bool(row.get("mask_triggered", row.get("should_purify", False)))]
    first_mean = _mean(first_vals)
    deployment_mean = _mean(deployment_vals)
    overhead_ratio = None
    if first_mean is not None and first_mean > 1e-9 and deployment_mean is not None:
        overhead_ratio = deployment_mean / first_mean

    return {
        "suite": suite,
        "run_id_note": run_id_note,
        "num_steps": len(rows),
        "mask_rate": len(masked_rows) / len(rows) if rows else None,
        "mean_first_policy_forward_ms": first_mean,
        "mean_defense_controller_ms": _mean(_values(rows, "defense_controller_ms")),
        "mean_purification_ms": _mean(_values(rows, "purification_ms")),
        "mean_second_policy_forward_ms": _mean(_values(rows, "second_policy_forward_ms")),
        "mean_deployment_total_ms": deployment_mean,
        "median_deployment_total_ms": _median(deployment_vals),
        "p95_deployment_total_ms": _percentile(deployment_vals, 0.95),
        "mean_masked_step_total_ms": _mean(_values(masked_rows, "deployment_total_ms")),
        "mean_unmasked_step_total_ms": _mean(_values(unmasked_rows, "deployment_total_ms")),
        "overhead_ratio_vs_first_forward_mean": overhead_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize runtime_metrics.jsonl files into CSV.")
    parser.add_argument("--input_glob", required=True, help="Glob for one or more runtime_metrics.jsonl files.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    rows = _read_runtime_rows(paths)
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summaries = [
        _summarize_group(suite, run_id_note, group_rows)
        for (suite, run_id_note), group_rows in sorted(grouped.items())
    ]

    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Wrote {len(summaries)} summary rows from {len(paths)} files to {args.output_csv}")


if __name__ == "__main__":
    main()

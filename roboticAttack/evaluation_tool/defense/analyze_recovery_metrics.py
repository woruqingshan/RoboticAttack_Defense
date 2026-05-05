#!/usr/bin/env python3
"""Aggregate recovery metrics JSONL logs into CSV summaries.

Example:
    python evaluation_tool/defense/analyze_recovery_metrics.py \
      --input_dir experiments/logs/libero_spatial_metrics \
      --output_dir experiments/logs/libero_spatial_metrics/csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def _mean(rows: Iterable[Dict[str, Any]], key: str):
    vals = [float(row[key]) for row in rows if _is_number(row.get(key))]
    if not vals:
        return None
    return float(np.mean(vals))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    preferred = [
        "record_type",
        "task_suite_name",
        "run_id_note",
        "task_id",
        "task_description",
        "episode",
        "episode_id",
        "step",
        "success",
        "success_rate",
    ]
    fieldnames = [k for k in preferred if k in fields] + [k for k in fields if k not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _load_jsonl_files(input_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(input_dir.rglob("metrics*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    row.setdefault("source_file", str(file))
                    rows.append(row)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Failed to parse {file}:{line_no}: {exc}")
    return rows


def _episode_summary_from_frames(frame_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        key = (
            row.get("task_suite_name"),
            row.get("run_id_note"),
            row.get("task_id"),
            row.get("episode_id", row.get("episode")),
        )
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    mean_keys = [
        "mask_patch_iou",
        "mask_patch_recall",
        "mask_patch_precision",
        "mask_area_ratio",
        "selected_roi_patch_iou",
        "selected_roi_patch_recall",
        "pam_adv",
        "pam_def",
        "pam_clean",
        "pam_reduction",
        "topk_attn_iou_patch_adv",
        "topk_attn_iou_patch_def",
        "attn_center_dist_patch_adv",
        "attn_center_dist_patch_def",
        "action_l2_adv_to_clean",
        "action_l2_def_to_clean",
        "nar_l2",
        "nar_xyz",
    ]
    for (_suite, _run, _task, _ep), rows in groups.items():
        rows_sorted = sorted(rows, key=lambda r: int(r.get("step", 0)))
        first = rows_sorted[0]
        first_masked = next((r for r in rows_sorted if bool(r.get("masked"))), None)
        summary = {
            "record_type": "episode_summary_from_frames",
            "task_suite_name": first.get("task_suite_name"),
            "run_id_note": first.get("run_id_note"),
            "task_id": first.get("task_id"),
            "task_description": first.get("task_description"),
            "episode": first.get("episode"),
            "episode_id": first.get("episode_id"),
            "masked_frames": int(sum(1 for r in rows_sorted if bool(r.get("masked")))),
            "first_lock_step": first_masked.get("step") if first_masked else None,
            "first_roi_box": first_masked.get("roi_box") if first_masked else None,
            "frame_count": len(rows_sorted),
        }
        for key in mean_keys:
            summary[f"mean_{key}"] = _mean(rows_sorted, key)
        out.append(summary)
    return out


def _task_summary(episode_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        key = (row.get("task_suite_name"), row.get("run_id_note"), row.get("task_id"))
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    mean_episode_keys = [
        "masked_frames",
        "mean_mask_patch_iou",
        "mean_mask_patch_recall",
        "mean_mask_area_ratio",
        "mean_pam_adv",
        "mean_pam_def",
        "mean_pam_reduction",
        "mean_action_l2_adv_to_clean",
        "mean_action_l2_def_to_clean",
        "mean_nar_l2",
        "mean_nar_xyz",
    ]
    for (_suite, _run, _task), rows in groups.items():
        first = rows[0]
        success_vals = [1.0 if bool(r.get("success")) else 0.0 for r in rows if "success" in r]
        summary = {
            "record_type": "task_summary",
            "task_suite_name": first.get("task_suite_name"),
            "run_id_note": first.get("run_id_note"),
            "task_id": first.get("task_id"),
            "task_description": first.get("task_description"),
            "episode_count": len(rows),
            "success_rate": float(np.mean(success_vals)) if success_vals else None,
        }
        for key in mean_episode_keys:
            summary[key] = _mean(rows, key)
        out.append(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate recovery metrics JSONL logs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing metrics*.jsonl files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for CSV outputs.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = _load_jsonl_files(input_dir)
    frame_rows = [r for r in rows if r.get("record_type") == "frame_metrics"]
    episode_rows = [r for r in rows if r.get("record_type") == "episode_summary"]
    if not episode_rows and frame_rows:
        episode_rows = _episode_summary_from_frames(frame_rows)
    task_rows = _task_summary(episode_rows)

    _write_csv(output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(output_dir / "episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "task_metrics.csv", task_rows)

    print(
        f"Wrote {len(frame_rows)} frame rows, {len(episode_rows)} episode rows, "
        f"{len(task_rows)} task rows to {output_dir}"
    )


if __name__ == "__main__":
    main()


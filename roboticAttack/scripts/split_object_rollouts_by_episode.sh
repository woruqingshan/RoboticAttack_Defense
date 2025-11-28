#!/bin/bash

set -e

# Usage:
#   bash scripts/split_object_rollouts_by_episode.sh [--date YYYY_MM_DD] [--dry-run] [--dst3-double-underscore]
#     --date: which date folder under rollouts to split (default: 2025_11_16)
#     --dry-run: preview actions without moving files
#     --dst3-double-underscore: use object__clean instead of object_clean
#
# This script splits object rollouts into three buckets by episode index:
#   1-100  -> libero_object/author_uada_object/<DATE>
#   101-200-> libero_object/object_xy_24_24/<DATE>
#   201-300-> libero_object_clean/object_clean(<or object__clean>)/<DATE>

DATE="2025_11_16"
DRY_RUN=0
DST3_DOUBLE_UNDERSCORE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      DATE="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --dst3-double-underscore)
      DST3_DOUBLE_UNDERSCORE=1; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

SRC_BASE="/home/zifeng/siyuan/code/roboticAttack/rollouts"
SRC="${SRC_BASE}/libero_object/${DATE}"

DST1="${SRC_BASE}/libero_object/author_uada_object/${DATE}"
DST2="${SRC_BASE}/libero_object/object_xy_24_24/${DATE}"
if [[ $DST3_DOUBLE_UNDERSCORE -eq 1 ]]; then
  DST3="${SRC_BASE}/libero_object_clean/object__clean/${DATE}"
else
  DST3="${SRC_BASE}/libero_object_clean/object_clean/${DATE}"
fi

if [[ ! -d "$SRC" ]]; then
  echo "Source directory not found: $SRC"
  exit 1
fi

mkdir -p "$DST1" "$DST2" "$DST3"

echo "[Config]"
echo "  DATE: $DATE"
echo "  SRC:  $SRC"
echo "  DST1: $DST1   (episodes 1-100)"
echo "  DST2: $DST2   (episodes 101-200)"
echo "  DST3: $DST3   (episodes 201-300)"
echo "  DRY_RUN: $DRY_RUN"
echo

count_before_src=$(ls -1 "$SRC"/*.mp4 2>/dev/null | wc -l || true)
count_before_dst1=$(ls -1 "$DST1"/*.mp4 2>/dev/null | wc -l || true)
count_before_dst2=$(ls -1 "$DST2"/*.mp4 2>/dev/null | wc -l || true)
count_before_dst3=$(ls -1 "$DST3"/*.mp4 2>/dev/null | wc -l || true)

echo "[Before] counts:"
echo "  SRC : $count_before_src"
echo "  DST1: $count_before_dst1"
echo "  DST2: $count_before_dst2"
echo "  DST3: $count_before_dst3"
echo

shopt -s nullglob
moved1=0; moved2=0; moved3=0; skipped=0
for f in "$SRC"/*.mp4; do
  # extract episode number like ...--episode=123--...
  ep=$(basename "$f" | grep -oP '(?<=--episode=)\d+')
  if [[ -z "$ep" ]]; then
    echo "Skip (no episode found): $f"
    ((skipped++))
    continue
  fi

  if   (( ep >= 1   && ep <= 100 )); then
    echo "Move ep=$ep -> DST1: $(basename "$f")"
    if [[ $DRY_RUN -eq 0 ]]; then mv "$f" "$DST1"/; fi
    ((moved1++))
  elif (( ep >= 101 && ep <= 200 )); then
    echo "Move ep=$ep -> DST2: $(basename "$f")"
    if [[ $DRY_RUN -eq 0 ]]; then mv "$f" "$DST2"/; fi
    ((moved2++))
  elif (( ep >= 201 && ep <= 300 )); then
    echo "Move ep=$ep -> DST3: $(basename "$f")"
    if [[ $DRY_RUN -eq 0 ]]; then mv "$f" "$DST3"/; fi
    ((moved3++))
  else
    echo "Skip (episode out of expected range): $f"
    ((skipped++))
  fi
done

echo
echo "[Summary]"
echo "  Planned moves -> DST1: $moved1, DST2: $moved2, DST3: $moved3, Skipped: $skipped"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run only. No files moved. Re-run without --dry-run to apply."
  exit 0
fi

count_after_src=$(ls -1 "$SRC"/*.mp4 2>/dev/null | wc -l || true)
count_after_dst1=$(ls -1 "$DST1"/*.mp4 2>/dev/null | wc -l || true)
count_after_dst2=$(ls -1 "$DST2"/*.mp4 2>/dev/null | wc -l || true)
count_after_dst3=$(ls -1 "$DST3"/*.mp4 2>/dev/null | wc -l || true)

echo
echo "[After] counts:"
echo "  SRC : $count_after_src"
echo "  DST1: $count_after_dst1"
echo "  DST2: $count_after_dst2"
echo "  DST3: $count_after_dst3"

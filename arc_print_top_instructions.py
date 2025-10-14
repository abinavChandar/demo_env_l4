

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print the top-3 instruction sets for each ARC task based on candidates_summary.csv.

Selection:
  - Rank by mean_cell_accuracy (descending).
  - When there are ties at a score level, randomly shuffle the tied candidates (seeded).
  - Continue through descending score levels until 3 candidates are chosen.

It then loads the candidates JSON for the task and prints each chosen instruction text.

Inputs:
  --scores-glob: glob to find per-task 'candidates_summary.csv' files
  --candidates-root: root folder containing <split>/<task_id>.json (the candidates)
  --split: training or evaluation
  --seed: RNG seed for tie-breaking
  --max-tasks: cap how many tasks to print (0 = all)
  --shuffle: shuffle task order before limiting
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------- Utilities ----------------

@dataclass
class CandidateScore:
    candidate_index: int
    mean_cell_accuracy: float


def parse_summary_csv(path: Path) -> List[CandidateScore]:
    out: List[CandidateScore] = []
    with path.open("r", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                ci = int(row["candidate_index"])
                m = float(row["mean_cell_accuracy"])
                out.append(CandidateScore(candidate_index=ci, mean_cell_accuracy=m))
            except Exception:
                continue
    return out


def find_task_id_from_summary(summary_path: Path) -> str:
    """
    Try to infer task_id from a typical path like:
      .../<task_id>_allc_out/<task_id>.candidates_summary.csv
    Fallback: use the stem before '.candidates_summary'.
    """
    name = summary_path.name
    stem = summary_path.stem  # e.g. 'fcc82909.candidates_summary'
    task_id = stem.replace(".candidates_summary", "")
    # If directory ends with '_allc_out', it usually contains the task_id prefix
    parent = summary_path.parent.name
    if parent.endswith("_allc_out"):
        possible = parent[:-9]  # strip '_allc_out'
        if possible:
            task_id = possible
    return task_id


def read_candidates(candidates_root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    cpath = candidates_root / split / f"{task_id}.json"
    with cpath.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cands = data.get("candidates")
    if not isinstance(cands, list):
        raise ValueError(f"No candidates list in {cpath}")
    return cands


def extract_instruction_text(cand: Dict[str, Any]) -> Optional[str]:
    """
    Robustly pull instruction-like text from common fields.
    """
    # Common direct string fields
    for k in ("instructions", "instruction", "text", "content", "nl", "prompt", "instruction_text"):
        v = cand.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Bullet/step lists of strings
    for k in ("bullets", "lines", "steps", "instructions_list"):
        v = cand.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(x.strip() for x in v if x.strip())
            if joined:
                return joined

    # OpenAI-style messages array (prefer assistant messages from the end)
    msgs = cand.get("messages")
    if isinstance(msgs, list) and msgs:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            c = (isinstance(m, dict) and m.get("content"))
            if isinstance(c, str) and c.strip():
                return c.strip()

    # Nested variants
    for k in ("initial", "revised", "best", "final", "candidate", "draft"):
        sub = cand.get(k)
        if isinstance(sub, dict):
            for kk in ("instructions", "text", "content", "nl"):
                v = sub.get(kk)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for kk in ("bullets", "lines", "steps"):
                v = sub.get(kk)
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    joined = "\n".join(x.strip() for x in v if x.strip())
                    if joined:
                        return joined
    return None


def pick_topk_with_tie_shuffle(scores: List[CandidateScore], k: int, rng: random.Random) -> List[int]:
    """
    Group by mean_cell_accuracy (descending). Within a score group, shuffle candidates randomly
    (using 'rng') to break ties. Continue selecting until 'k' indices are picked or items run out.
    """
    if not scores:
        return []
    # Build mapping score -> [candidate_index...]
    buckets: Dict[float, List[int]] = {}
    for s in scores:
        buckets.setdefault(s.mean_cell_accuracy, []).append(s.candidate_index)

    chosen: List[int] = []
    for score in sorted(buckets.keys(), reverse=True):
        group = buckets[score][:]
        rng.shuffle(group)
        for ci in group:
            if len(chosen) >= k:
                break
            chosen.append(ci)
        if len(chosen) >= k:
            break
    return chosen


def find_summary_files(scores_glob: str) -> List[Tuple[str, Path]]:
    """
    Return list of (task_id, path_to_candidates_summary.csv) using the provided glob.
    Deduplicates by task_id, keeping the first occurrence.
    """
    raw: List[Tuple[str, Path]] = []
    for p in glob.glob(scores_glob, recursive=True):
        path = Path(p)
        if path.name.endswith("candidates_summary.csv"):
            tid = find_task_id_from_summary(path)
            raw.append((tid, path))

    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for tid, p in raw:
        if tid not in seen:
            seen.add(tid)
            uniq.append((tid, p))
    return uniq


# ---------------- Main logic ----------------

def main():
    ap = argparse.ArgumentParser(description="Print top-3 instruction sets per ARC task (with random tie-breaking).")
    ap.add_argument("--scores-glob", type=str, required=True,
                    help="Glob to find candidates_summary.csv files (e.g. /content/debug_multi/**/*candidates_summary.csv)")
    ap.add_argument("--candidates-root", type=Path, default=Path("/content/outputs"),
                    help="Root dir containing <split>/<task_id>.json with 'candidates'")
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaking")
    ap.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks to print (0 = no limit)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle task order before limiting")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    found = find_summary_files(args.scores_glob)
    if not found:
        print(f"[top3] No candidates_summary.csv files found with glob: {args.scores_glob}")
        sys.exit(0)

    if args.shuffle:
        rng.shuffle(found)
    if args.max_tasks and args.max_tasks > 0:
        found = found[:args.max_tasks]

    print(f"[top3] Found {len(found)} tasks to process.\n")

    for task_id, summary_path in found:
        try:
            scores = parse_summary_csv(summary_path)
            if not scores:
                print(f"=== Task {task_id} ===")
                print(f"(no rows in {summary_path})\n")
                continue

            chosen = pick_topk_with_tie_shuffle(scores, k=3, rng=rng)
            if not chosen:
                print(f"=== Task {task_id} ===")
                print("(no candidates selected)\n")
                continue

            # Load candidates JSON for this task
            try:
                cands = read_candidates(args.candidates_root, args.split, task_id)
            except Exception as e:
                print(f"=== Task {task_id} ===")
                print(f"ERROR: cannot read candidates JSON for {task_id}: {e}\n")
                continue

            print(f"=== Task {task_id} ===")
            print(f"summary: {summary_path}")
            # Show quick leaderboard top few (optional for context)
            top_sorted = sorted(scores, key=lambda s: s.mean_cell_accuracy, reverse=True)[:6]
            top_line = ", ".join(f"{s.candidate_index}:{s.mean_cell_accuracy:.2f}%" for s in top_sorted)
            print(f"leaderboard (top): {top_line}")
            print(f"selected indices: {chosen}\n")

            # Print each selected instruction set
            for rank, ci in enumerate(chosen, 1):
                txt = None
                if 0 <= ci < len(cands):
                    txt = extract_instruction_text(cands[ci])
                heading = f"[#{rank}] candidate {ci}"
                print(heading)
                print("-" * len(heading))
                if txt and txt.strip():
                    print(txt.strip())
                else:
                    print("<no instruction text found>")
                print()  # blank line between entries

        except Exception as e:
            print(f"=== Task {task_id} ===")
            print(f"ERROR: {e}\n")

    print("[top3] Done.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate synthesized Python programs on ARC training pairs.

For each task:
  • Read synthesized JSON with {"python_code": "...", "dsl_operations": [...]}
  • exec() the module; require def transform(grid) -> List[List[int]]
  • Run over all training pairs; compute cell accuracy vs ground-truth outputs
  • Print per-pair results and a per-task mean accuracy
  • (optional) write per-pair CSV and a summary CSV

Expected synthesized file layout:
  --synth-dir/<task_id>.synth.json   (keys: python_code, dsl_operations, ...)

Env (if you later extend to re-call a model): OPENAI_BASE_URL / OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------- ARC helpers ----------------

def is_valid_grid(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    w = None
    for row in obj:
        if not isinstance(row, list) or not row:
            return False
        if w is None:
            w = len(row)
        if len(row) != w:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True

def cell_accuracy(pred: Optional[List[List[int]]], truth: List[List[int]]) -> float:
    if pred is None:
        return 0.0
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]):
        return 0.0
    h, w = len(truth), len(truth[0])
    match = 0
    for i in range(h):
        for j in range(w):
            if pred[i][j] == truth[i][j]:
                match += 1
    return 100.0 * match / (h * w)

def load_arc_pairs(root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    f = root / split / f"{task_id}.json"
    with f.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = data.get("train") or data.get("training") or data.get("pairs")
    if not pairs or not isinstance(pairs, list):
        raise ValueError(f"No training pairs in {f}")
    # validate
    out: List[Dict[str, Any]] = []
    for p in pairs:
        x = p["input"]; y = p["output"]
        if not (is_valid_grid(x) and is_valid_grid(y)):
            raise ValueError(f"Invalid grids in {f}")
        out.append({"input": x, "output": y})
    return out


# ---------------- Synth loading & execution ----------------

def load_synth_json(synth_dir: Path, task_id: str) -> Dict[str, Any]:
    fp = synth_dir / f"{task_id}.synth.json"
    with fp.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{fp} is not a JSON object")
    if "python_code" not in data or not isinstance(data["python_code"], str):
        raise ValueError(f"{fp} missing 'python_code' string")
    return data

def build_transform_callable(python_code: str):
    """
    Execute the provided module text and return the transform function.
    We deliberately give a minimal global namespace. Your synthesized code
    must be pure Python (std lib only) and define def transform(grid): ...
    """
    module = types.ModuleType("synth_module")
    # Minimal builtins; you can relax if your code needs more
    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "sum": sum,
        "abs": abs, "enumerate": enumerate, "map": map, "filter": filter,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "all": all, "any": any, "zip": zip,
    }
    module.__dict__["__builtins__"] = safe_builtins
    exec(python_code, module.__dict__)
    fn = module.__dict__.get("transform")
    if not callable(fn):
        raise ValueError("Synthesized module does not define callable transform(grid)")
    return fn


# ---------------- Evaluation ----------------

def evaluate_task(task_id: str,
                  root: Path, split: str,
                  synth_dir: Path,
                  print_grids: bool) -> Tuple[float, List[Dict[str, Any]]]:
    pairs = load_arc_pairs(root, split, task_id)
    synth = load_synth_json(synth_dir, task_id)
    transform = build_transform_callable(synth["python_code"])

    per_pair_rows: List[Dict[str, Any]] = []
    total = 0.0

    print(f"\n=== Task {task_id} ===")
    for i, p in enumerate(pairs):
        x = p["input"]; y = p["output"]
        try:
            pred = transform([row[:] for row in x])  # defensive copy
        except Exception as e:
            pred = None
            note = f"runtime error: {e!r}"
        else:
            note = "ok" if is_valid_grid(pred) else "invalid grid"
            if not is_valid_grid(pred):
                pred = None
        acc = cell_accuracy(pred, y)
        total += acc

        # Print per pair
        print(f"[Pair {i}] ACC={acc:.2f}%  note={note}")
        if print_grids:
            def g2s(g): return "\n".join(" ".join(str(v) for v in row) for row in g)
            print("INPUT:");  print(g2s(x))
            print("TARGET:"); print(g2s(y))
            if pred is None:
                print("PRED  : <None>")
            else:
                print("PRED  :"); print(g2s(pred))
            print("-" * 60)

        per_pair_rows.append({
            "task_id": task_id,
            "pair_index": i,
            "cell_accuracy": round(acc, 4),
            "note": note,
            "pred": pred,
            "input": x,
            "target": y,
        })

    mean_acc = total / len(pairs) if pairs else 0.0
    print(f"[Task {task_id}] MEAN_CELL_ACC = {mean_acc:.2f}%")
    return mean_acc, per_pair_rows


# ---------------- CLI & main ----------------

def discover_tasks_from_synth(synth_dir: Path) -> List[str]:
    task_ids = []
    for p in synth_dir.glob("*.synth.json"):
        task_ids.append(p.stem.replace(".synth", ""))  # in case name was like "<tid>.synth"
        # If you always name as "<tid>.synth.json", the replace is harmless.
    return sorted(set(task_ids))

def main():
    ap = argparse.ArgumentParser(description="Evaluate synthesized ARC programs on training pairs.")
    ap.add_argument("--root", type=Path, required=True, help="ARC data root (contains <split>/<task_id>.json)")
    ap.add_argument("--split", type=str, choices=["training","evaluation"], default="training")
    ap.add_argument("--synth-dir", type=Path, required=True, help="Dir with <task_id>.synth.json files")
    ap.add_argument("--task-ids", nargs="*", default=None, help="Optional explicit list of task ids to eval")
    ap.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks (0 = all)")
    ap.add_argument("--print-grids", action="store_true", help="Print input/target/pred grids per pair")
    ap.add_argument("--out-dir", type=Path, default=None, help="If set, write per-pair CSV and summary CSV here")
    args = ap.parse_args()

    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = discover_tasks_from_synth(args.synth_dir)

    if args.max_tasks and args.max_tasks > 0:
        task_ids = task_ids[:args.max_tasks]

    if not task_ids:
        print(f"No tasks found to evaluate from {args.synth_dir}", file=sys.stderr)
        sys.exit(1)

    # Optional writers
    per_pair_writer = None
    summary_writer = None
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_pair_fp = args.out_dir / "per_pair.csv"
        summary_fp = args.out_dir / "summary.csv"
        per_pair_writer = csv.writer(per_pair_fp.open("w", newline="", encoding="utf-8"))
        per_pair_writer.writerow(["task_id","pair_index","cell_accuracy","note","input","target","pred"])
        summary_writer = csv.writer(summary_fp.open("w", newline="", encoding="utf-8"))
        summary_writer.writerow(["task_id","mean_cell_accuracy"])

    for tid in task_ids:
        mean_acc, rows = evaluate_task(tid, args.root, args.split, args.synth_dir, args.print_grids)
        if per_pair_writer:
            for r in rows:
                per_pair_writer.writerow([
                    r["task_id"], r["pair_index"], f"{r['cell_accuracy']:.4f}", r["note"],
                    json.dumps(r["input"], separators=(",", ":")),
                    json.dumps(r["target"], separators=(",", ":")),
                    json.dumps(r["pred"], separators=(",", ":")) if r["pred"] is not None else "",
                ])
        if summary_writer:
            summary_writer.writerow([tid, f"{mean_acc:.4f}"])

    print("\n[eval] Done.")

if __name__ == "__main__":
    main()


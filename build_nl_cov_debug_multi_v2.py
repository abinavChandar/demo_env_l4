
from __future__ import annotations
from pathlib import Path
import argparse, json, re
import pandas as pd

# ---------- File discovery ----------

TASK_DIR_RE = re.compile(r"^(?P<task>[0-9a-fA-F]+)_allc_out$")

def discover_tasks(in_dir: Path):
    tasks = {}
    for sub in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        m = TASK_DIR_RE.match(sub.name)
        if not m:
            continue
        task = m.group("task")
        jsonl = sub / f"{task}.per_pair.jsonl"
        csv   = sub / f"{task}.per_pair.csv"
        cand_csv = sub / f"{task}.candidates_summary.csv"
        tasks[task] = {"dir": sub, "per_pair_jsonl": jsonl if jsonl.exists() else None,
                       "per_pair_csv": csv if csv.exists() else None,
                       "candidates_csv": cand_csv if cand_csv.exists() else None}
    return tasks

# ---------- Parsing helpers ----------

def _norm_acc(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
        if f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))
    except Exception:
        s = str(v).strip().lower()
        if s in {"true","pass","passed","exact"}: return 1.0
        if s in {"false","fail","failed","nonexact","no"}: return 0.0
        return None

def _pick(d: dict, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def parse_per_pair_jsonl(path: Path, task: str):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pair_id = _pick(obj, ["pair_id","pair","pair_idx","example_id","idx"])
            # Case A: list under "scores" or "candidates"
            for key in ("scores","candidates","results","per_candidate"):
                if key in obj:
                    block = obj[key]
                    if isinstance(block, dict):  # mapping id -> acc
                        for cid, acc in block.items():
                            acc = _norm_acc(acc)
                            rows.append({"task_id": task, "pair_id": str(pair_id), "nl_set_id": str(cid), "acc": acc if acc is not None else 0.0})
                        return rows
                    if isinstance(block, list):
                        for ent in block:
                            cid = _pick(ent, ["nl_set_id","instruction_set_id","candidate_id","candidate","name","id"])
                            acc = _pick(ent, ["acc","accuracy","cell_accuracy","score","exact","pass"])
                            acc = _norm_acc(acc)
                            rows.append({"task_id": task, "pair_id": str(pair_id), "nl_set_id": str(cid), "acc": acc if acc is not None else 0.0})
                        return rows
            # Case B: flattened fields like cand_id_*, acc_*
            for k,v in obj.items():
                if k.startswith("cand_") and isinstance(v, (int,float,bool,str)):
                    acc = _norm_acc(v)
                    cid = k[len("cand_"):]
                    rows.append({"task_id": task, "pair_id": str(pair_id), "nl_set_id": str(cid), "acc": acc if acc is not None else 0.0})
    return rows

CAND_KEYS   = ["nl_set_id","instruction_set_id","candidate_id","candidate","name","id"]
ACC_KEYS    = ["acc","accuracy","cell_accuracy","score","exact","pass"]
PAIR_KEYS   = ["pair_id","pair","pair_idx","example_id","idx"]

def parse_per_pair_csv(path: Path, task: str):
    df = pd.read_csv(path)
    # Try to identify columns
    cand_col = next((c for c in CAND_KEYS if c in df.columns), None)
    if cand_col is None:
        # guess first column holding string-ish ids besides pair
        for c in df.columns:
            if c.lower() in [x.lower() for x in PAIR_KEYS]:
                continue
            if df[c].dtype == object:
                cand_col = c; break
    pair_col = next((p for p in PAIR_KEYS if p in df.columns), None)
    acc_col  = next((a for a in ACC_KEYS if a in df.columns), None)
    if pair_col is None:
        # many per_pair.csv pivot shapes: wide format with candidates as columns
        # handle: first col = pair_id, rest columns = candidate ids with accuracy values
        pair_col = df.columns[0]
        wide_cols = [c for c in df.columns[1:]]
        wide = df.melt(id_vars=[pair_col], value_vars=wide_cols, var_name="nl_set_id", value_name="acc")
        wide["task_id"] = task
        wide["pair_id"] = wide[pair_col].astype(str)
        wide["acc"] = pd.to_numeric(wide["acc"], errors="coerce").fillna(0.0).clip(0.0,1.0)
        return wide[["task_id","pair_id","nl_set_id","acc"]].to_dict("records")
    # tidy format
    out = []
    for _,r in df.iterrows():
        pair_id = r[pair_col]
        # either direct candidate col, or we explode wide if missing
        if cand_col and acc_col:
            nl = r[cand_col]
            acc = r[acc_col]
            out.append({"task_id": task, "pair_id": str(pair_id), "nl_set_id": str(nl),
                        "acc": float(acc) if pd.notna(acc) else 0.0})
        else:
            # wide format: every column except pair_col is a candidate with numeric acc
            for c in df.columns:
                if c == pair_col: continue
                acc = r[c]
                out.append({"task_id": task, "pair_id": str(pair_id), "nl_set_id": str(c),
                            "acc": float(acc) if pd.notna(acc) else 0.0})
    return out

# ---------- Coverage builders ----------

def build_coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = df.pivot_table(index="nl_set_id", columns="pair_id", values="acc", aggfunc="mean", fill_value=0.0)
    return m.sort_index(axis=0).sort_index(axis=1)

def build_pass_matrix(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cov = build_coverage_matrix(df)
    return (cov >= threshold).astype(int)

def compute_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    mean_acc = df.groupby("nl_set_id")["acc"].mean().rename("mean_acc")
    pass_df = df.assign(pass_bool=(df["acc"] >= threshold).astype(int))
    pass_rate = pass_df.groupby("nl_set_id")["pass_bool"].mean().rename("exact_pass_rate")
    passed_pairs = pass_df[pass_df["pass_bool"] == 1].groupby("nl_set_id")["pair_id"].nunique().rename("pairs_covered")
    pairs_total = df.groupby("nl_set_id")["pair_id"].nunique().rename("pairs_total")
    summary = pd.concat([mean_acc, pass_rate, passed_pairs, pairs_total], axis=1).fillna(0.0)
    return summary.sort_values(by=["exact_pass_rate","mean_acc","pairs_covered"], ascending=[False, False, False]).reset_index()

def greedy_cover(df: pd.DataFrame, threshold: float):
    pairs = set(df["pair_id"].unique().tolist())
    pass_map = (
        df.assign(pass_bool=(df["acc"] >= threshold).astype(int))
          .groupby("nl_set_id")
          .apply(lambda g: set(g.loc[g["pass_bool"] == 1, "pair_id"].unique().tolist()))
          .to_dict()
    )
    uncovered = set(pairs)
    chosen, steps = [], []
    while uncovered:
        best, best_gain = None, -1
        for nl, covered in pass_map.items():
            if nl in chosen: continue
            gain = len(covered & uncovered)
            if gain > best_gain:
                best_gain, best = gain, nl
        if best is None or best_gain <= 0:
            break
        chosen.append(best)
        newly = sorted(list(pass_map[best] & uncovered))
        uncovered -= pass_map[best]
        steps.append({"chosen": best, "newly_covered": newly, "remaining_uncovered": len(uncovered)})
    info = {"covered_pairs": sorted(list(pairs - uncovered)),
            "uncovered_pairs": sorted(list(uncovered)),
            "steps": steps}
    return chosen, info

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Build NL coverage matrices from /debug_multi per_pair.{jsonl,csv}")
    ap.add_argument("--in-dir", required=True, help="Root /debug_multi directory.")
    ap.add_argument("--out", required=True, help="Output directory for matrices/summary/plan.")
    ap.add_argument("--task-id", default=None, help="Specific task id (e.g., 305b1341). Omit for all tasks.")
    ap.add_argument("--threshold", type=float, default=1.0, help="Pass threshold on acc.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(in_dir)
    if not tasks:
        raise SystemExit(f"No <task>_allc_out directories found under {in_dir}")

    if args.task_id:
        tasks = {k:v for k,v in tasks.items() if str(k) == str(args.task_id)}
        if not tasks:
            raise SystemExit(f"Task {args.task_id} not found under {in_dir}")

    for task, info in tasks.items():
        rows = []
        if info["per_pair_jsonl"]:
            rows.extend(parse_per_pair_jsonl(info["per_pair_jsonl"], task))
        if not rows and info["per_pair_csv"]:
            rows.extend(parse_per_pair_csv(info["per_pair_csv"], task))

        if not rows:
            print(f"[WARN] No rows parsed for task {task} (looked in {info['dir']}). Skipping.")
            continue

        df = pd.DataFrame(rows)
        # Clean types
        df["acc"] = pd.to_numeric(df["acc"], errors="coerce").fillna(0.0).clip(0.0,1.0)
        df["pair_id"] = df["pair_id"].astype(str)
        df["nl_set_id"] = df["nl_set_id"].astype(str)

        cov = build_coverage_matrix(df)
        pas = build_pass_matrix(df, args.threshold)
        summary = compute_summary(df, args.threshold)
        chosen, info_cov = greedy_cover(df, args.threshold)

        cov_path = out_dir / f"{task}_nl_coverage_matrix.csv"
        pas_path = out_dir / f"{task}_nl_pass_matrix_thr{args.threshold}.csv"
        sum_path = out_dir / f"{task}_nl_summary_thr{args.threshold}.csv"
        plan_path = out_dir / f"{task}_nl_greedy_cover_thr{args.threshold}.json"

        cov.to_csv(cov_path)
        pas.to_csv(pas_path)
        summary.to_csv(sum_path, index=False)
        plan_path.write_text(json.dumps({"chosen_order": chosen, "info": info_cov}, indent=2))

        print(f"[OK] {task}: NL sets={cov.shape[0]} Pairs={cov.shape[1]}")
        if info_cov["uncovered_pairs"]:
            print(f"    Uncovered pairs @thr={args.threshold}: {info_cov['uncovered_pairs']}")
        else:
            print(f"    All pairs covered (greedy).")
        print(f"    Saved -> {cov_path.name}, {pas_path.name}, {sum_path.name}, {plan_path.name}")
        
if __name__ == "__main__":
    main()

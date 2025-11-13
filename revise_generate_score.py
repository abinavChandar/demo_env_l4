#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Set

Grid = List[List[int]]

# ---------------- logging / fs ----------------
def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def candidate_dir(scores_root: str, task_id: str) -> str:
    return os.path.join(scores_root, f"{task_id}_allc_out")

def list_candidates(scores_root: str, task_id: str) -> List[str]:
    return sorted(
        glob.glob(
            os.path.join(candidate_dir(scores_root, task_id),
                         f"{task_id}.cand_*.transform.nl.py")
        )
    )

# ---------------- discovery ----------------
def discover_tasks(instr_root: str, recursive: bool) -> List[Tuple[str, str]]:
    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(glob.glob(os.path.join(instr_root, pattern), recursive=recursive))
    out: List[Tuple[str, str]] = []
    for p in paths:
        base = os.path.basename(p)
        if base.lower().endswith(".json"):
            out.append((base[:-5], p))
    return out

def unpack_task(task: Any) -> Tuple[str, str]:
    if isinstance(task, dict):
        return str(task["task_id"]), str(task["instr_json"])
    if isinstance(task, (tuple, list)) and len(task) >= 2:
        return str(task[0]), str(task[1])
    raise TypeError(f"Bad task shape: {task!r}")

# ---------------- ARC I/O ----------------
def find_task_json(arc_root: str, task_id: str) -> str:
    direct = os.path.join(arc_root, f"{task_id}.json")
    if os.path.isfile(direct):
        return direct
    hits = glob.glob(os.path.join(arc_root, "**", f"{task_id}.json"), recursive=True)
    if hits:
        return sorted(hits, key=lambda p: (p.count(os.sep), p))[0]
    raise FileNotFoundError(f"ARC json not found for {task_id}")

def load_pairs(task_json_path: str) -> List[Tuple[Grid, Grid]]:
    with open(task_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs: List[Tuple[Grid, Grid]] = []
    if isinstance(data, dict) and isinstance(data.get("train"), list):
        for ex in data["train"]:
            if "input" in ex and "output" in ex:
                pairs.append((ex["input"], ex["output"]))
    elif isinstance(data, list):
        for ex in data:
            if "input" in ex and "output" in ex:
                pairs.append((ex["input"], ex["output"]))
    if not pairs:
        raise ValueError("No training pairs in task json")
    return pairs

# ---------------- safe import + timeout ----------------
def _import_module_from_path(mod_path: str):
    spec = importlib.util.spec_from_file_location("cand_mod", mod_path)
    if not spec or not spec.loader:
        raise ImportError(f"spec load failed for {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _worker_transform(mod_path: str, grid: Grid, q: mp.Queue):
    try:
        mod = _import_module_from_path(mod_path)
        if not hasattr(mod, "transform"):
            q.put(("error", "No transform()"))
            return
        out = mod.transform(grid)
        q.put(("ok", out))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))

def run_with_timeout(mod_path: str, grid: Grid, t: float) -> Tuple[bool, Optional[Grid], Optional[str]]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_transform, args=(mod_path, grid, q))
    p.daemon = True
    p.start()
    p.join(t)
    if p.is_alive():
        p.terminate()
        p.join(0.1)
        return (False, None, f"Timeout({t:.2f}s)")
    try:
        status, payload = q.get_nowait()
    except Exception:
        return (False, None, "No result returned")
    if status == "ok":
        return (True, payload, None)
    else:
        return (False, None, payload)

# ---------------- metrics ----------------
def same_shape(A: Grid, B: Grid) -> bool:
    return len(A) == len(B) and all(len(a) == len(b) for a, b in zip(A, B))

def per_cell_acc(pred: Grid, gold: Grid) -> float:
    if not same_shape(pred, gold):
        return 0.0
    tot = len(gold) * (len(gold[0]) if gold else 0)
    ok = sum(
        1
        for r in range(len(gold))
        for c in range(len(gold[0]))
        if pred[r][c] == gold[r][c]
    )
    return (ok / tot) if tot else 0.0

def exact_match(pred: Grid, gold: Grid) -> bool:
    return same_shape(pred, gold) and pred == gold

# ---------------- scoring ----------------
def score_candidate_on_pairs(
    mod_path: str, pairs: List[Tuple[Grid, Grid]], timeout_sec: float
) -> Dict[str, Any]:
    per_pair: List[Dict[str, Any]] = []
    n_exact = 0
    acc_sum = 0.0
    t0 = time.time()
    for i, (inp, gold) in enumerate(pairs):
        ok, out, err = run_with_timeout(mod_path, inp, timeout_sec)
        rec: Dict[str, Any] = {"pair": i}
        if not ok:
            rec.update({"status": "error", "error": err})
            per_pair.append(rec)
            continue
        if not isinstance(out, list) or (out and not isinstance(out[0], list)):
            rec.update(
                {"status": "error", "error": "transform returned non-grid"}
            )
            per_pair.append(rec)
            continue
        em = exact_match(out, gold)
        pa = per_cell_acc(out, gold)
        if em:
            n_exact += 1
        acc_sum += pa
        rec.update(
            {"status": "ok", "exact": em, "per_cell_acc": round(pa, 6)}
        )
        per_pair.append(rec)
    n = len(pairs)
    return {
        "candidate_path": mod_path,
        "pairs": per_pair,
        "exact_total": n_exact,
        "exact_rate": (n_exact / n) if n else 0.0,
        "mean_per_cell_acc": (acc_sum / n) if n else 0.0,
        "elapsed_sec": round(time.time() - t0, 3),
    }

def score_task_candidates(
    task_json_path: str, cand_paths: List[str], timeout: float, print_stdout: bool = True
) -> Dict[str, Any]:
    out = {
        "task_json": task_json_path,
        "timeout": timeout,
        "results": [],
        "errors": [],
        "ok": False,
        "best": None,
    }
    try:
        pairs = load_pairs(task_json_path)
    except Exception as e:
        out["errors"].append(
            {"candidate_path": "<init>", "error": f"{type(e).__name__}: {e}"}
        )
        return out

    best = None
    for pth in cand_paths:
        try:
            rep = score_candidate_on_pairs(pth, pairs, timeout)
            out["results"].append(rep)

            if print_stdout:
                print("\n" + "=" * 80)
                print(f"[CANDIDATE] {os.path.basename(pth)}")
                for pp in rep["pairs"]:
                    i = pp["pair"]
                    if pp["status"] == "ok":
                        em = "✓" if pp["exact"] else " "
                        print(
                            f"  pair {i:>2}: acc={pp['per_cell_acc']:.6f}  exact=[{em}]"
                        )
                    else:
                        print(f"  pair {i:>2}: ERROR: {pp['error']}")
                print(
                    f"  --- summary: exact_total={rep['exact_total']}/{len(pairs)}  "
                    f"mean_acc={rep['mean_per_cell_acc']:.4f}  time={rep['elapsed_sec']:.3f}s"
                )
                print("=" * 80)

            ma = float(rep.get("mean_per_cell_acc") or 0.0)
            if (best is None) or (ma > best["mean_acc"]):
                best = {
                    "path": pth,
                    "mean_acc": ma,
                    "exact_total": int(rep.get("exact_total", 0)),
                }
        except Exception as e:
            out["errors"].append(
                {"candidate_path": pth, "error": f"{type(e).__name__}: {e}"}
            )

    out["best"] = best
    out["ok"] = len(out["results"]) > 0
    return out

# ---------------- generator ----------------
def build_task_cmd(
    per_task_script: str,
    scores_root: str,
    task_id: str,
    instr_json: str,
    dsl_module: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    gen_timeout: float,
    workers: int,
    seed: Optional[int],
    arc_root: Optional[str],
    shots: int,
    feedback_file: Optional[str],
    verbose: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        per_task_script,
        "--scores-root",
        scores_root,
        "--task-id",
        task_id,
        "--instr-json",
        instr_json,
        "--dsl-module",
        dsl_module,
        "--api-base",
        api_base,
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--retries",
        str(retries),
        "--timeout",
        str(gen_timeout),
        "--workers",
        str(workers),
        "--no-print",
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    if arc_root:
        cmd += ["--arc-root", arc_root, "--shots", str(shots)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # feedback_file is currently not wired through to llamagenerate (to avoid arg errors)
    if verbose:
        cmd += ["--verbose"]
    return cmd

def regenerate_candidates(
    task_id: str, instr_json: str, args, temperature: float, feedback_path: Optional[str]
) -> int:
    out_dir = candidate_dir(args.scores_root, task_id)
    ensure_dir(out_dir)
    cmd = build_task_cmd(
        args.per_task_script,
        args.scores_root,
        task_id,
        instr_json,
        args.dsl_module,
        args.api_base,
        (args.api_key or ""),
        args.model,
        temperature,
        args.max_tokens,
        args.retries,
        args.gen_timeout,
        args.workers,
        args.seed,
        args.arc_root,
        args.shots,
        feedback_path,
        args.verbose,
    )
    log("INFO", f"{task_id}: generate → {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(f"[{task_id}] {line.rstrip()}", flush=True)
        p.wait()
    except Exception as e:
        log("ERROR", f"{task_id}: generator exception {e}")
    return len(list_candidates(args.scores_root, task_id))

# ---------------- feedback assembly ----------------
def build_feedback_text(scored: Dict[str, Any]) -> str:
    """
    Compile a concise, LLM-friendly error summary from scoring results.
    """
    msgs: List[str] = []
    for rep in (scored.get("results") or []):
        name = os.path.basename(rep.get("candidate_path", ""))
        errs: List[str] = []
        for pp in rep.get("pairs", []):
            if pp.get("status") != "ok":
                e = str(pp.get("error", "")).strip()
                if e:
                    errs.append(e)
        if errs:
            # dedupe & truncate
            uniq: List[str] = []
            seen: Set[str] = set()
            for e in errs:
                k = e.split(":")[0].strip()
                if k not in seen:
                    seen.add(k)
                    uniq.append(e[:140])
            msgs.append(f"- {name}: " + "; ".join(uniq))
    return "\n".join(msgs[:24])  # keep it short

# ---------------- GLOBAL counters across all tasks ----------------
GLOBAL_SEEN_CANDIDATES: Set[str] = set()
GLOBAL_GOOD_CANDIDATES: Set[str] = set()
GLOBAL_LOCK = threading.Lock()

# ---------------- revision loop ----------------
def task_revision_loop(task: Any, args) -> Dict[str, Any]:
    task_id, instr_json = unpack_task(task)
    tdir = candidate_dir(args.scores_root, task_id)
    ensure_dir(tdir)

    best_mean: Optional[float] = None
    best_path: Optional[str] = None

    rounds = int(args.max_rounds)
    base_temp = float(args.temperature)
    temp_step = float(args.temp_step)
    temp_max = float(args.temp_max)

    task_json_path = find_task_json(args.arc_root, task_id)

    # Track per-candidate error history for this task (to mark FIXED)
    candidate_had_error_before: Dict[str, bool] = {}

    for r in range(1, rounds + 1):
        curT = min(base_temp + (r - 1) * temp_step, temp_max)

        # Feedback file from previous round (if any)
        feedback_path = os.path.join(tdir, "regen_feedback.txt")
        feedback_for_round = feedback_path if os.path.isfile(feedback_path) else None

        log(
            "INFO",
            f"{task_id}: round {r}/{rounds} — generate @ T={curT:.2f} "
            f"(feedback={'yes' if feedback_for_round else 'no'})",
        )
        regenerate_candidates(task_id, instr_json, args, curT, feedback_for_round)

        cand_paths = list_candidates(args.scores_root, task_id)
        if not cand_paths:
            log("WARN", f"{task_id}: no candidates found; continuing")
            continue

        scored = score_task_candidates(
            task_json_path, cand_paths, args.score_timeout, print_stdout=True
        )

        # Per-round summary counts
        total = len(cand_paths)
        clean = 0
        errored = 0

        candidate_good_now: Dict[str, bool] = {}
        fixed_this_round: List[str] = []

        # Round-level global stats snapshot (will be updated as we loop)
        global_good_all = 0
        global_total_all = 0

        for rep in (scored.get("results") or []):
            cand_path = rep.get("candidate_path", "")
            pairs = rep.get("pairs", [])

            any_error = any(pp.get("status") != "ok" for pp in pairs)
            all_ok = (len(pairs) > 0) and (not any_error)
            mean_acc = float(rep.get("mean_per_cell_acc") or 0.0)

            # "good" = no errors on any pair AND meets min-acc (if specified)
            is_good = all_ok and (
                args.min_acc is None or mean_acc >= float(args.min_acc)
            )

            # track best
            if best_mean is None or mean_acc > best_mean:
                best_mean = mean_acc
                best_path = cand_path

            if any_error:
                errored += 1
            else:
                clean += 1

            # error-history logic for FIXED marker
            prev_had_err = candidate_had_error_before.get(cand_path, False)
            now_had_err = any_error

            if prev_had_err and all_ok and not now_had_err:
                fixed_this_round.append(cand_path)

            candidate_had_error_before[cand_path] = prev_had_err or now_had_err
            candidate_good_now[cand_path] = is_good

            # ---- GLOBAL counters across all tasks ----
            if cand_path:
                with GLOBAL_LOCK:
                    GLOBAL_SEEN_CANDIDATES.add(cand_path)
                    if is_good:
                        GLOBAL_GOOD_CANDIDATES.add(cand_path)
                    global_total_all = len(GLOBAL_SEEN_CANDIDATES)
                    global_good_all = len(GLOBAL_GOOD_CANDIDATES)

        # round summary (per task)
        log(
            "INFO",
            f"{task_id}: round {r} summary → total={total}, clean={clean}, errored={errored}",
        )

        # Green “FIXED” marks for candidates that recovered this round
        for pth in fixed_this_round:
            short = os.path.basename(pth)
            print(
                f"[FIXED] \033[92m✔\033[0m {task_id}: candidate recovered and now fully OK → {short}",
                flush=True,
            )

        # Task-level snapshot: how many candidates for this task are GOOD vs QUEUE
        task_total = len(candidate_good_now)
        task_good = sum(1 for ok in candidate_good_now.values() if ok)
        task_queue = task_total - task_good

        log(
            "INFO",
            f"[SUMMARY] {task_id}: task-level → good_ok={task_good}/{task_total} | in_queue={task_queue}",
        )

        # Global snapshot (across ALL tasks seen so far)
        if global_total_all > 0:
            log(
                "INFO",
                f"[SUMMARY] GLOBAL: good_ok={global_good_all}/{global_total_all} across all tasks so far",
            )

        # Write/update feedback for the NEXT round (if needed)
        fb_text = build_feedback_text(scored)
        if fb_text:
            with open(feedback_path, "w", encoding="utf-8") as f:
                f.write(
                    "Avoid these failure modes when generating transform():\n"
                    "- Do NOT use direct indexing grid[r][c] or attributes; use DSL ops only.\n"
                    "- Do NOT return generators/iterators; return a concrete grid (List[List[int]]).\n"
                    "- Only call known dsl_* primitives available in dsl_primitives.\n"
                    "- Ensure shape matches expected outputs; use DSL ops to crop/resize as needed.\n\n"
                    + fb_text
                    + "\n"
                )

        # Stop early if all are clean and (optional) best meets min-acc
        all_clean = (errored == 0 and clean == total and total > 0)
        meets_min = (args.min_acc is None) or (
            best_mean is not None and best_mean >= float(args.min_acc)
        )
        if all_clean and meets_min:
            log("INFO", f"{task_id}: all candidates clean; stopping early.")
            break

    return {
        "task_id": task_id,
        "status": ("ok" if best_mean is not None else "empty"),
        "best_mean_acc": float(best_mean) if best_mean is not None else 0.0,
        "best_path": best_path,
        "rounds": rounds,
    }

# ---------------- CLI ----------------
def main() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--instr-root", required=True)
    ap.add_argument("--arc-root", required=True)
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument(
        "--per-task-script",
        default="llama_generate_all_transforms_for_task_parallel.py",
    )
    ap.add_argument("--dsl-module", default="dsl_primitives")
    ap.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
    )
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument(
        "--model", default=os.environ.get("LLM_MODEL", "llama-3.1-8b-instruct")
    )
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--temp-step", type=float, default=0.05)
    ap.add_argument("--temp-max", type=float, default=0.60)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--gen-timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--score-timeout", type=float, default=4.0)
    ap.add_argument("--min-acc", type=float, default=None)
    ap.add_argument("--max-rounds", type=int, default=3)

    ap.add_argument("--tasks-workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only-tasks", type=str, default=None)

    args = ap.parse_args()
    if not hasattr(args, "timeout") or args.timeout is None:
        args.timeout = args.score_timeout

    tasks = discover_tasks(args.instr_root, args.recursive)
    if args.only_tasks:
        allow = {t.strip() for t in args.only_tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t[0] in allow]
    if not tasks:
        log("ERROR", "No tasks discovered")
        sys.exit(1)

    log("INFO", f"Discovered {len(tasks)} task(s). Starting revision loops…")
    ensure_dir(os.path.join(args.scores_root, "telemetry"))

    results: List[Dict[str, Any]] = []

    def run_task(task: Any) -> Dict[str, Any]:
        try:
            return task_revision_loop(task, args)
        except Exception as e:
            tid = None
            try:
                tid, _ = unpack_task(task)
            except Exception:
                pass
            return {
                "task_id": tid or "<unknown>",
                "status": "exception",
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }

    with ThreadPoolExecutor(max_workers=args.tasks_workers) as pool:
        futs = [pool.submit(run_task, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res.get("status") == "ok":
                log(
                    "OK",
                    f"{res['task_id']}: best_mean_acc={res.get('best_mean_acc',0):.4f} "
                    f":: {os.path.basename(res.get('best_path') or '')}",
                )
            else:
                log(
                    "WARN",
                    f"{res.get('task_id')}: status={res.get('status')} "
                    f"{res.get('error','')}",
                )

    summary_path = os.path.join(
        args.scores_root, "telemetry", f"revise_loop_summary_{int(time.time())}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    log("INFO", f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
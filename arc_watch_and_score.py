

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
arc_watch_and_score.py — Watch a candidates directory and score tasks as their
instruction JSONs appear. Works alongside your existing arc_multi_task_score.py.

Why this approach:
- You can run arc_instructions_generate.py in one process (or Colab cell).
- Run this watcher in another; it will begin scoring each task as soon as its
  /content/outputs/training/<task_id>.json file is *stable* on disk.

It launches arc_multi_task_score.py as a subprocess per *batch* of newly ready tasks,
so you don’t have to modify your scorer. Batching keeps overhead low and reuses
your scorer’s own concurrency knobs.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

def infer_task_id(p: Path) -> str | None:
    return p.stem if p.suffix.lower() == ".json" else None

def stable_new_files(glob_pat: str, known: Set[str], min_age: float, size_stable_secs: float) -> List[Tuple[str, Path]]:
    """
    Returns list of (task_id, path) for files not in 'known' whose size is stable
    for >= size_stable_secs and whose mtime is older than min_age seconds.
    """
    now = time.time()
    files = [Path(p) for p in glob.glob(glob_pat)]
    out: List[Tuple[str, Path]] = []
    for f in files:
        tid = infer_task_id(f)
        if not tid or tid in known:
            continue
        try:
            st1 = f.stat()
        except FileNotFoundError:
            continue
        # wait until file is old enough and size isn't changing
        too_new = now - st1.st_mtime < min_age
        if too_new:
            continue
        # size stability check
        size1 = st1.st_size
        time.sleep(size_stable_secs)
        try:
            st2 = f.stat()
        except FileNotFoundError:
            continue
        size2 = st2.st_size
        if size1 == size2 and size2 > 0:
            out.append((tid, f))
    return out

async def run_scorer_once(
    task_ids: List[str],
    args: argparse.Namespace,
) -> int:
    """
    Invoke arc_multi_task_score.py for a *batch* of task_ids, passing through
    your scoring flags. Returns the subprocess return code.
    """
    cmd = [
        sys.executable, str(args.scorer_script),
        "--root", str(args.root),
        "--split", args.split,
        "--model", args.model,
        "--candidates-file", str(args.candidates_file),
        "--candidates-layout-per-task",
        "--out-root", str(args.out_root),
        "--pair-concurrency", str(args.pair_concurrency),
        "--candidate-concurrency", str(args.candidate_concurrency),
        "--task-concurrency", str(args.task_concurrency),
        "--global-request-concurrency", str(args.global_request_concurrency),
    ]
    # Optional flags
    if args.guided_json:
        cmd.append("--guided-json")
    if args.guided_regex:
        cmd.append("--guided-regex")
    if args.no_response_format:
        cmd.append("--no-response-format")
    if args.enforce_shape:
        cmd.append("--enforce-shape")
    if args.mismatch_policy:
        cmd += ["--mismatch-policy", args.mismatch_policy]

    # Provide the tasks
    cmd += ["--task-ids"] + task_ids

    print(f"\n[watch] Launching scorer for {len(task_ids)} task(s): {' '.join(task_ids)}")
    print("[watch] CMD:", shlex.join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    # Stream output live
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", errors="ignore"))
    rc = await proc.wait()
    print(f"[watch] scorer finished (rc={rc}) for batch of {len(task_ids)} tasks.")
    return rc

async def main():
    ap = argparse.ArgumentParser(description="Watch candidates dir and score tasks as JSONs appear.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    ap.add_argument("--candidates-glob", type=str, default="/content/outputs/training/*.json")
    ap.add_argument("--candidates-file", type=Path, default=Path("/content/outputs/training/placeholder.json"),
                   help="Placeholder (ignored when --candidates-layout-per-task is used by scorer).")
    ap.add_argument("--scorer-script", type=Path, default=Path("arc_multi_task_score.py"),
                   help="Path to your scorer script.")
    ap.add_argument("--model", type=str, required=True)

    # Scorer concurrency flags
    ap.add_argument("--pair-concurrency", type=int, default=8)
    ap.add_argument("--candidate-concurrency", type=int, default=1)
    ap.add_argument("--task-concurrency", type=int, default=2)
    ap.add_argument("--global-request-concurrency", type=int, default=16)

    # Scorer decoding/constraints
    ap.add_argument("--guided-json", action="store_true")
    ap.add_argument("--guided-regex", action="store_true")
    ap.add_argument("--no-response-format", action="store_true")
    ap.add_argument("--enforce-shape", action="store_true")
    ap.add_argument("--mismatch-policy", type=str, choices=["zero","scan","crop","pad","fit","best"], default=None)

    # Watcher behavior
    ap.add_argument("--out-root", type=Path, default=Path("/content/debug_multi"))
    ap.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between scans.")
    ap.add_argument("--min-file-age", type=float, default=1.0, help="File must be at least this old (s).")
    ap.add_argument("--size-stable-secs", type=float, default=0.5, help="Size must be stable for this many seconds.")
    ap.add_argument("--batch-size", type=int, default=8, help="Max new tasks per scoring batch.")
    ap.add_argument("--max-parallel-batches", type=int, default=1, help="How many scorer batches to run concurrently.")
    ap.add_argument("--limit-tasks", type=int, default=0, help="Stop after processing this many tasks (0 = infinite).")
    args = ap.parse_args()

    seen: Set[str] = set()
    inflight: Set[str] = set()
    total_done = 0

    sem_batches = asyncio.Semaphore(max(1, args.max_parallel_batches))
    running_batches: Set[asyncio.Task] = set()

    async def launch_batch(tids: List[str]):
        nonlocal total_done
        async with sem_batches:
            for tid in tids:
                inflight.add(tid)
            rc = await run_scorer_once(tids, args)
            for tid in tids:
                inflight.discard(tid)
                seen.add(tid)
                total_done += 1
            if rc != 0:
                print(f"[watch] WARNING: scorer returned rc={rc} for batch {tids}")

    try:
        print(f"[watch] Watching {args.candidates_glob} ... (Ctrl+C to stop)")
        while True:
            # Discover new, stable files
            new_ready = stable_new_files(args.candidates_glob, known=seen|inflight,
                                         min_age=args.min_file_age, size_stable_secs=args.size_stable_secs)
            if new_ready:
                # Batch up to batch-size
                tids = [tid for tid, _ in new_ready][:args.batch_size]
                task = asyncio.create_task(launch_batch(tids))
                running_batches.add(task)
                task.add_done_callback(lambda t: running_batches.discard(t))
            # Stop if limit reached
            if args.limit_tasks and total_done >= args.limit_tasks:
                print(f"[watch] Reached limit-tasks={args.limit_tasks}. Exiting.")
                break
            await asyncio.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[watch] KeyboardInterrupt — waiting for running batches to finish...")
        if running_batches:
            await asyncio.gather(*list(running_batches), return_exceptions=True)
        print("[watch] Done.")

if __name__ == "__main__":
    asyncio.run(main())

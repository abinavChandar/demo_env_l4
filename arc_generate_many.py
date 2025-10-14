
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch many arc_instructions_generate.py runs in parallel.

- Discovers task_ids from --tasks-glob (filenames in your ARC data split)
- Runs up to --task-concurrency tasks at once
- For each task, calls your existing arc_instructions_generate.py with --task-id
- Streams logs from each subprocess
"""

from __future__ import annotations
import argparse, asyncio, glob, os, shlex, sys
from pathlib import Path
from typing import List

def discover_task_ids(tasks_glob: str) -> List[str]:
    ids = []
    for p in glob.glob(tasks_glob):
        path = Path(p)
        if path.suffix.lower() == ".json":
            ids.append(path.stem)
    # stable order
    ids.sort()
    return ids

async def run_one(tid: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, str(args.generator_script),
        "--root", str(args.root),
        "--split", args.split,
        "--task-id", tid,
        "--num-candidates", str(args.num_candidates),
        "--outputs", str(args.out_dir),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.model:           cmd += ["--model", args.model]
    if args.temperature is not None: cmd += ["--temperature", str(args.temperature)]
    if args.seed is not None:        cmd += ["--seed", str(args.seed)]
    if args.timeout:         cmd += ["--timeout", str(args.timeout)]
    if args.retries:         cmd += ["--retries", str(args.retries)]
    if args.backoff:         cmd += ["--backoff", str(args.backoff)]
    if args.use_chat:        cmd.append("--use-chat")
    if args.ctx_limit:       cmd += ["--ctx-limit", str(args.ctx_limit)]
    if args.ctx_margin:      cmd += ["--ctx-margin", str(args.ctx_margin)]
    if args.min_completion:  cmd += ["--min-completion", str(args.min_completion)]
    if args.compact:         cmd.append("--compact")
    if args.ascii_style:     cmd.append("--ascii-style")
    if args.overwrite:       cmd.append("--overwrite")
    if args.candidate_tries: cmd += ["--candidate-tries", str(args.candidate_tries)]

    print(f"[gen-many] START {tid}\nCMD: {shlex.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    assert proc.stdout
    async for line in proc.stdout:
        sys.stdout.write(f"[{tid}] {line.decode('utf-8', errors='ignore')}")
    rc = await proc.wait()
    print(f"[gen-many] END {tid} rc={rc}")
    return rc

async def main_async(args: argparse.Namespace) -> None:
    tids = discover_task_ids(args.tasks_glob)
    if args.max_tasks and args.max_tasks > 0:
        tids = tids[:args.max_tasks]
    print(f"[gen-many] Discovered {len(tids)} tasks from {args.tasks_glob}")
    if not tids:
        print("[gen-many] No tasks found."); return

    sem = asyncio.Semaphore(max(1, args.task_concurrency))
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    async def _bounded(tid: str):
        async with sem:
            return await run_one(tid, args)

    results = await asyncio.gather(*[asyncio.create_task(_bounded(t)) for t in tids])
    ok = sum(1 for r in results if r == 0)
    print(f"[gen-many] Done: {ok}/{len(results)} succeeded.")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concurrent launcher for arc_instructions_generate_many.py")
    p.add_argument("--generator-script", type=Path, default=Path("arc_instructions_generate_many.py"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", type=str, choices=["training","evaluation"], default="training")

    # discovery / limit / concurrency
    p.add_argument("--tasks-glob", type=str, required=True,
                   help='e.g. /content/ARC-AGI-2/data/training/*.json (task_id from filename)')
    p.add_argument("--max-tasks", type=int, default=0, help="Process at most N tasks (0 = no limit)")
    p.add_argument("--task-concurrency", type=int, default=2, help="Parallel generator processes")

    # pass-through generator args
    p.add_argument("--model", type=str, default="")
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=0.6)
    p.add_argument("--use-chat", action="store_true")
    p.add_argument("--ctx-limit", type=int, default=None)
    p.add_argument("--ctx-margin", type=int, default=None)
    p.add_argument("--min-completion", type=int, default=None)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--ascii-style", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--candidate-tries", type=int, default=None)
    return p

def main():
    args = build_parser().parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()

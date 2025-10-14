
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARC — concurrent instruction-set generator across MANY tasks.

Generates N instruction candidates per task and writes
  out_dir/<task_id>.json  with schema:
  {
    "task_id": "...",
    "split": "...",
    "candidates": [{"instructions": "<text>"}, ...]
  }

Concurrency:
  • --task-concurrency: how many tasks in parallel
  • --candidate-concurrency: per task, how many candidates in parallel
  • --global-request-concurrency: cap total HTTP requests across everything

Safety:
  • atomic writes (tmp rename) so a watcher/scorer can pick files as soon as ready
  • optional per-candidate retries/backoff

Env:
  OPENAI_BASE_URL (e.g. http://127.0.0.1:8000/v1)
  OPENAI_API_KEY  (any non-empty string)
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import glob
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- ARC IO ----------------

def load_arc_training_pairs(root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    f = root / split / f"{task_id}.json"
    with f.open("r", encoding="utf-8") as fh:
        task = json.load(fh)
    pairs = task.get("train") or task.get("training") or task.get("pairs")
    if not pairs:
        raise ValueError(f"No training pairs in {f}")
    # Normalize to [{"input": [[...]], "output": [[...]]}, ...]
    norm: List[Dict[str, Any]] = []
    for p in pairs:
        if "input" in p and "output" in p:
            norm.append({"input": p["input"], "output": p["output"]})
    if not norm:
        raise ValueError(f"Bad pairs format in {f}")
    return norm

def collect_task_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.task_ids:
        ids.extend(args.task_ids)
    if args.task_file:
        ids.extend([t.strip() for t in Path(args.task_file).read_text(encoding="utf-8").splitlines() if t.strip()])
    if args.tasks_glob:
        for p in glob.glob(args.tasks_glob):
            path = Path(p)
            if path.suffix.lower() == ".json":
                ids.append(path.stem)
    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out and args.task_id:
        out = [args.task_id]
    if not out:
        raise SystemExit("No task ids. Use --task-ids, --task-file, --tasks-glob, or --task-id.")
    return out

# ---------------- LLM client ----------------

@dataclass
class GenResult:
    text: Optional[str]
    error: Optional[str]

class ChatClient:
    def __init__(self, session: aiohttp.ClientSession, *, timeout: int, retries: int, backoff: float):
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.key = os.environ.get("OPENAI_API_KEY", "sk")
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.backoff = backoff

    async def generate(self, *, model: str, system: str, user: str, max_tokens: int, temperature: float, seed: Optional[int], global_sem: asyncio.Semaphore) -> GenResult:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            payload["seed"] = int(seed)  # supported by vLLM & many servers

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                async with global_sem:
                    async with self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout) as resp:
                        body = await resp.text()
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}: {body[:300]}"
                else:
                    try:
                        data = json.loads(body)
                        ch = data.get("choices", [])
                        if ch:
                            return GenResult(text=ch[0].get("message", {}).get("content", ""), error=None)
                        last_err = "no choices"
                    except Exception as e:
                        last_err = f"decode error: {e} | body[:200]={body[:200]}"
            except Exception as e:
                last_err = f"request error: {e}"
            await asyncio.sleep(self.backoff * (attempt + 1))
        return GenResult(text=None, error=last_err)

# ---------------- Prompting ----------------

SYSTEM = (
    "You are an expert at the Abstraction and Reasoning Corpus (ARC). "
    "Given the training pairs (INPUT grid -> OUTPUT grid), infer a single, clear transformation rule. "
    "Return ONLY the instruction text. Do NOT include code fences or JSON."
)

def build_instruction_prompt(task_id: str, pairs: List[Dict[str, Any]], *, max_pairs: int = 6) -> str:
    # Include a few pairs to keep prompt compact if tasks have many examples
    subset = pairs[:max_pairs]
    lines = [f"TASK {task_id}: Infer the minimal rule that maps INPUT to OUTPUT for all training pairs.",
             "Training pairs:"]
    for idx, p in enumerate(subset):
        x = json.dumps(p["input"], separators=(",", ":"))
        y = json.dumps(p["output"], separators=(",", ":"))
        lines.append(f"- Pair {idx}:\n  INPUT={x}\n  OUTPUT={y}")
    lines.append(
        "\nReturn the instruction set as concise numbered steps or bullet points. "
        "Avoid generic heuristics; be specific about geometry, colors, repetition, mirroring, and resizing."
    )
    return "\n".join(lines)

# ---------------- Generation core ----------------

async def gen_candidates_for_task(
    client: ChatClient,
    args: argparse.Namespace,
    task_id: str,
    global_sem: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> None:
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.json"
    tmp_path = out_dir / f"{task_id}.json.tmp"

    pairs = load_arc_training_pairs(args.root, args.split, task_id)
    user_prompt = build_instruction_prompt(task_id, pairs, max_pairs=args.max_pairs_in_prompt)

    async def _one_candidate(ci: int) -> Dict[str, Any]:
        # small diversity via temperature + seed
        temp = args.temperature if args.temperature is not None else 0.2
        seed = None if args.seed is None else (args.seed + ci)
        res = await client.generate(
            model=args.model,
            system=SYSTEM,
            user=user_prompt,
            max_tokens=args.max_tokens,
            temperature=temp,
            seed=seed,
            global_sem=global_sem,
        )
        text = (res.text or "").strip()
        if not text:
            text = f"<empty; error={res.error}>"
        async with print_lock:
            print(f"[gen] {task_id} cand#{ci}: {'OK' if res.error is None else 'ERR'}")
        return {"instructions": text}

    sem_cands = asyncio.Semaphore(max(1, args.candidate_concurrency))
    async def _bounded(ci: int):
        async with sem_cands:
            return await _one_candidate(ci)

    tasks = [asyncio.create_task(_bounded(ci)) for ci in range(args.num_candidates)]
    candidates = await asyncio.gather(*tasks)

    # Atomic write
    payload = {"task_id": task_id, "split": args.split, "candidates": candidates}
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    wall = time.perf_counter() - t0
    async with print_lock:
        print(f"[gen] {task_id}: wrote {out_path}  (candidates={len(candidates)})  wall={wall:.2f}s")

# ---------------- Driver ----------------

async def main_async(args: argparse.Namespace):
    tids = collect_task_ids(args)
    # in main_async(args) after: tids = collect_task_ids(args)
    if args.shuffle:
        import random; random.shuffle(tids)
    if args.max_tasks and args.max_tasks > 0:
        tids = tids[:args.max_tasks]
    print(f"[gen] tasks={len(tids)} | task_conc={args.task_concurrency} | "
          f"cand_conc={args.candidate_concurrency} | global_req={args.global_request_concurrency}")

    print(f"[gen] tasks={len(tids)} | task_conc={args.task_concurrency} | cand_conc={args.candidate_concurrency} | global_req={args.global_request_concurrency}")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = ChatClient(session, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
        print_lock = asyncio.Lock()
        sem_tasks = asyncio.Semaphore(max(1, args.task_concurrency))
        global_sem = asyncio.Semaphore(max(1, args.global_request_concurrency))

        async def _run_tid(tid: str):
            async with sem_tasks:
                try:
                    await gen_candidates_for_task(client, args, tid, global_sem, print_lock)
                except Exception as e:
                    async with print_lock:
                        print(f"[gen] {tid}: ERROR {e}")

        await asyncio.gather(*[asyncio.create_task(_run_tid(t)) for t in tids])

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ARC concurrent instruction generator across many tasks.")
    # tasks
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    p.add_argument("--task-id", type=str)
    p.add_argument("--task-ids", nargs="*")
    p.add_argument("--task-file", type=Path)
    p.add_argument("--tasks-glob", type=str, help="e.g. /content/ARC-AGI-2/data/training/*.json (uses basenames as task_ids)")

    # LLM
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=0.6)

    # how many candidates per task
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--max-pairs-in-prompt", type=int, default=6)

    # in build_arg_parser()
    p.add_argument("--max-tasks", type=int, default=0,
                  help="Process at most N tasks (0 = no limit)")
    p.add_argument("--shuffle", action="store_true",
                  help="Shuffle discovered tasks before applying --max-tasks")


    # concurrency
    p.add_argument("--task-concurrency", type=int, default=2)
    p.add_argument("--candidate-concurrency", type=int, default=2)
    p.add_argument("--global-request-concurrency", type=int, default=16)

    # output
    p.add_argument("--out-dir", type=Path, required=True)
    return p

def main():
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARC — apply ALL candidate instruction sets to ALL training pairs across MANY tasks, concurrently.

Features:
  • Full logs per pair: INPUT / TARGET / RAW / PRED / ACC
  • Adaptive max_tokens based on output HxW
  • Strict JSON parse + lenient numeric-scan (shape-aware) recovery
  • JSONL/CSV per pair + per-candidate summary CSV per task
  • Guided JSON / regex and response_format hints (vLLM/OpenAI-compatible)

Scaling:
  • Run N tasks concurrently (--task-ids, --task-file, or --tasks-glob)
  • --task-concurrency to limit parallel tasks
  • --candidate-concurrency and --pair-concurrency per task
  • --global-request-concurrency caps total in-flight HTTP requests

NEW:
  • Task is marked SOLVED if ANY candidate gets 100.00% on ALL pairs
  • Per-task summary line with best candidate, parsed_ok, SOLVED/UNSOLVED, task wall-time
  • Final multi-task report: solved count, solved IDs, per-task best accuracies, overall wall-time
  • tasks_summary.csv written to --out-root (or current working directory)

Env:
  OPENAI_BASE_URL  (e.g. http://127.0.0.1:8000/v1)
  OPENAI_API_KEY   (any non-empty string)
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import csv
import glob as _glob
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- Minimal ARC helpers ----------

def is_valid_grid(obj: Any) -> bool:
    """True iff obj is a rectangular list[list[int]] with ints in 0..9 and >=1x1."""
    if not isinstance(obj, list) or not obj:
        return False
    if not all(isinstance(row, list) and row for row in obj):
        return False
    w = len(obj[0])
    if w == 0:
        return False
    for row in obj:
        if len(row) != w:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True

def grid_to_ascii(g: List[List[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)

def cell_accuracy(pred: Optional[List[List[int]]], truth: List[List[int]]) -> float:
    if pred is None:
        return 0.0
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]):
        return 0.0
    h, w = len(truth), len(truth[0])
    match = sum(1 for i in range(h) for j in range(w) if pred[i][j] == truth[i][j])
    return 100.0 * match / (h * w)

def try_extract_json_grid_strict(text: str) -> Tuple[Optional[List[List[int]]], str]:
    """Accept only clean JSON array-of-arrays of ints 0..9 with equal row lengths."""
    last_err = "uninitialized"
    try:
        obj = json.loads(text)
        if is_valid_grid(obj):
            return obj, "ok"
        last_err = "direct json is not a valid grid"
    except Exception as e:
        last_err = f"json.loads error: {e}"

    # Fallback: bracket-slice first outermost array
    try:
        s = text.strip()
        start = s.index("[")
        end = s.rindex("]") + 1
        snippet = s[start:end]
        obj2 = json.loads(snippet)
        if is_valid_grid(obj2):
            return obj2, "ok: bracket-slice"
        return None, "invalid grid after bracket-slice"
    except Exception as e:
        return None, f"parse_error: bracket-slice failed: {e} | prev={last_err}"

def try_extract_grid_lenient_with_shape(text: str, target_shape: Optional[Tuple[int,int]]) -> Tuple[Optional[List[List[int]]], str]:
    """
    Deterministic fallback when strict JSON parse fails.
    If target_shape is known, scan digits 0–9 in order and take the first H*W numbers row-major.
    Recovers from truncated commas/brackets *if* all digits are present.
    """
    if not target_shape:
        return None, "lenient needs target_shape"
    H, W = target_shape
    need = H * W
    nums = [int(m.group(0)) for m in re.finditer(r"[0-9]", text or "")]
    if len(nums) < need:
        return None, f"lenient: only {len(nums)}/{need} digits present"
    nums = nums[:need]
    grid = [nums[i*W:(i+1)*W] for i in range(H)]
    if not is_valid_grid(grid):
        return None, "lenient: constructed grid failed validation"
    return grid, "ok: lenient numeric-scan with enforce-shape"

# ---------- Task loading ----------

def load_arc_training_pairs(root: Path, split: str, task_id: str) -> List[Tuple[List[List[int]], List[List[int]]]]:
    f = root / split / f"{task_id}.json"
    with f.open("r", encoding="utf-8") as fh:
        task = json.load(fh)
    pairs = task.get("train") or task.get("training") or task.get("pairs") or []
    if not pairs:
        raise ValueError(f"No training pairs in {f}")
    out: List[Tuple[List[List[int]], List[List[int]]]] = []
    for p in pairs:
        x = p["input"]
        y = p["output"]
        if not is_valid_grid(x) or not is_valid_grid(y):
            raise ValueError("Input/output in file is not a valid grid")
        out.append((x, y))
    return out

# ---------- Instruction extraction (robust) ----------

def _dot_get(obj: Any, path: str) -> Any:
    if not path:
        return None
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur

def _candidate_text_fields(cand: Dict[str, Any]) -> Dict[str, str]:
    previews: Dict[str, str] = {}
    def add(name: str, val: Any):
        if isinstance(val, str) and val.strip():
            previews[name] = val.strip()[:160].replace("\n", " ")
        elif isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            joined = "\n".join(x.strip() for x in val if x.strip())
            if joined:
                previews[name] = joined[:160].replace("\n", " ")
    for k in ("instructions", "instruction", "text", "content", "nl", "prompt", "instruction_set", "instruction_text"):
        if k in cand:
            add(k, cand[k])
    for k in ("bullets", "lines", "steps", "instructions_list"):
        if k in cand:
            add(k, cand[k])
    msgs = cand.get("messages")
    if isinstance(msgs, list) and msgs:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and "content" in m:
                add(f"messages.{i}.content", m["content"])
                break
    for k in ("initial", "revised", "best", "final", "candidate", "draft"):
        if k in cand and isinstance(cand[k], dict):
            for kk in ("instructions", "text", "content", "nl"):
                if kk in cand[k]:
                    add(f"{k}.{kk}", cand[k][kk])
            for kk in ("bullets", "lines", "steps"):
                if kk in cand[k]:
                    add(f"{k}.{kk}", cand[k][kk])
    return previews

def _normalize_candidates(data: Any) -> List[Dict[str, Any]]:
    seq: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for key in ("candidates", "items", "results", "list", "data"):
            if key in data and isinstance(data[key], list):
                seq = [x for x in data[key] if isinstance(x, dict)]
                break
        if not seq and data and all(isinstance(v, dict) for v in data.values()):
            seq = list(data.values())
    elif isinstance(data, list):
        seq = [x for x in data if isinstance(x, dict)]
    return seq

def _extract_text_from_candidate(cand: Dict[str, Any], instructions_key: str) -> Optional[str]:
    if instructions_key:
        val = _dot_get(cand, instructions_key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            joined = "\n".join(x.strip() for x in val if x.strip())
            if joined:
                return joined
    for k in ("instructions", "instruction", "text", "content", "nl", "prompt", "instruction_text"):
        v = cand.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("bullets", "lines", "steps", "instructions_list"):
        v = cand.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(x.strip() for x in v if x.strip())
            if joined:
                return joined
    msgs = cand.get("messages")
    if isinstance(msgs, list) and msgs:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                c = m["content"].strip()
                if c:
                    return c
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                c = m["content"].strip()
                if c:
                    return c
    for k in ("initial", "revised", "best", "final", "candidate", "draft"):
        sub = cand.get(k)
        if isinstance(sub, dict):
            for kk in ("instructions", "text", "content", "nl"):
                v = sub.get(kk)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for kk in ("bullets", "lines", "steps"):
                v = sub.get(kk)
                if v and all(isinstance(x, str) for x in v):
                    joined = "\n".join(x.strip() for x in v if x.strip())
                    if joined:
                        return joined
    return None

def dump_candidates_only(candidates_file: Path, limit: int = 10) -> None:
    with candidates_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    seq = _normalize_candidates(data)
    if not seq:
        print("No candidates found.")
        return
    n = min(limit, len(seq))
    print(f"Found {len(seq)} candidates. Showing previews for first {n}:")
    for i in range(n):
        cand = seq[i]
        previews = _candidate_text_fields(cand)
        if not previews:
            print(f"  [{i}] <no text-like fields detected>")
        else:
            for k, v in previews.items():
                print(f"  [{i}] {k} => {v}")

# ---------- Prompt + client ----------

APPLY_SYSTEM = (
    "You are an ARC grid transformer. APPLY the given instructions to the INPUT grid.\n"
    "Return ONLY the OUTPUT grid as JSON like [[0,1],[2,3]] — integers 0–9, no prose, no code fences.\n"
    "If the rule resizes, compute the new dimensions accordingly; if not, preserve size."
)

def build_apply_prompt(task_id: str,
                       instructions: str,
                       input_grid: List[List[int]],
                       target_shape: Optional[Tuple[int, int]]) -> str:
    tgt = ""
    if target_shape:
        th, tw = target_shape
        tgt = (
            "\nREQUIREMENT: The OUTPUT grid MUST have EXACT shape "
            f"H_out={th}, W_out={tw}. Use only digits 0–9."
        )
    prompt = (
        f"TASK {task_id}\n"
        f"INSTRUCTIONS:\n{instructions.strip()}\n\n"
        f"INPUT (H={len(input_grid)}, W={len(input_grid[0])}):\n"
        f"{json.dumps(input_grid, separators=(',', ':'))}\n"
        f"{tgt}\n"
        f"{APPLY_SYSTEM}\n"
    )
    return prompt

@dataclass
class GenResult:
    text: Optional[str]
    error: Optional[str]

class AsyncCompatClient:
    """Chat completions client with optional vLLM guided_json / guided_regex and response_format."""
    def __init__(self,
                 session: aiohttp.ClientSession,
                 timeout: int = 180,
                 max_tokens: int = 256,
                 retries: int = 4,
                 backoff: float = 0.6,
                 use_guided_json: bool = False,
                 use_guided_regex: bool = False,
                 use_response_format: bool = True) -> None:
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff
        self.use_guided_json = use_guided_json
        self.use_guided_regex = use_guided_regex
        self.use_response_format = use_response_format
        self._session = session

    def _json_schema(self) -> Dict[str, Any]:
        return {
            "name": "arc_grid",
            "schema": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 9},
                    "minItems": 1
                },
                "minItems": 1
            },
            "strict": True,
        }

    def _regex_pattern(self) -> str:
        return r"\[\s*(\[\s*(\d\s*,\s*)*\d\s*\]\s*,\s*)*(\[\s*(\d\s*,\s*)*\d\s*\])\s*\]"

    def _payload(self) -> Dict[str, Any]:
        return {}

    async def generate(self, model: str, system: str, user: str, max_tokens: int, *, use_guided_json: bool, use_guided_regex: bool, use_response_format: bool, global_sem: asyncio.Semaphore) -> GenResult:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if use_guided_json:
            payload["guided_json"] = self._json_schema()["schema"]
        elif use_guided_regex:
            payload["guided_regex"] = self._regex_pattern()
        if use_response_format and not use_guided_json:
            payload["response_format"] = {"type": "json_schema", "json_schema": self._json_schema()}

        last_err: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                async with global_sem:
                    async with self._session.post(self.url, headers=headers, json=payload, timeout=self.timeout) as resp:
                        txt = await resp.text()
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}: {txt[:400]}"
                else:
                    try:
                        data = json.loads(txt)
                        ch = data.get("choices", [])
                        if ch:
                            msg = ch[0].get("message", {})
                            return GenResult(text=msg.get("content", ""), error=None)
                        last_err = "no choices returned"
                    except Exception as e:
                        last_err = f"decode error: {e} | body[:200]={txt[:200]}"
            except Exception as e:
                last_err = f"request error: {e}"
            await asyncio.sleep(self.backoff * (attempt + 1))
        return GenResult(text=None, error=last_err)

# ---------- Async writer helpers ----------

class RowWriter:
    """Async writer that consumes rows and writes JSONL/CSV without blocking the main tasks."""
    def __init__(self, jsonl_path: Path, csv_path: Path):
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.queue: asyncio.Queue = asyncio.Queue()
        self._jsonl_fh = None
        self._csv_fh = None
        self._csv_writer = None
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_fh = self.jsonl_path.open("w", encoding="utf-8")
        self._csv_fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_writer.writerow(["candidate_index","pair_index","cell_accuracy","note","input","target","pred"])
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.queue.put(None)  # sentinel
        if self._task:
            await self._task
        if self._jsonl_fh: self._jsonl_fh.close()
        if self._csv_fh: self._csv_fh.close()

    async def _run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            ci = item["candidate_index"]
            r = item["row"]
            self._jsonl_fh.write(json.dumps({"candidate_index": ci, **item["row"]}, ensure_ascii=False) + "\n")
            self._csv_writer.writerow([
                ci,
                r["pair_index"],
                f"{r['cell_accuracy']:.4f}",
                r["note"],
                json.dumps(r["input"], separators=(",", ":")),
                json.dumps(r["target"], separators=(",", ":")),
                json.dumps(r["pred"], separators=(",", ":")) if r["pred"] is not None else "",
            ])

# ---------- Per-candidate / per-pair evaluation ----------

async def eval_one_pair(
    client: AsyncCompatClient,
    args: argparse.Namespace,
    task_id: str,
    candidate_index: int,
    pair_index: int,
    x: List[List[int]],
    y: List[List[int]],
    instructions: str,
    print_lock: asyncio.Lock,
    global_sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    target_shape = (len(y), len(y[0])) if args.enforce_shape else None
    prompt = build_apply_prompt(task_id, instructions, x, target_shape)

    # Adaptive token budget: ~4 tokens per cell + 64 headroom
    if target_shape:
        H_out, W_out = target_shape
    else:
        H_out, W_out = len(x), len(x[0])
    per_pair_max_tokens = max(args.max_tokens, (H_out * W_out) * 4 + 64)

    res = await client.generate(
        model=args.model,
        system=APPLY_SYSTEM,
        user=prompt,
        max_tokens=per_pair_max_tokens,
        use_guided_json=args.guided_json,
        use_guided_regex=args.guided_regex,
        use_response_format=(not args.no_response_format),
        global_sem=global_sem,
    )

    raw = res.text if res.text is not None else f"<None>  ERROR={res.error}"
    pred, note = (None, "no text") if res.text is None else try_extract_json_grid_strict(res.text)
    if pred is None:
        pred2, note2 = try_extract_grid_lenient_with_shape(res.text or "", target_shape)
        if pred2 is not None:
            pred, note = pred2, note2

    acc = cell_accuracy(pred, y)

    # PRINT (serialize to avoid interleaving)
    async with print_lock:
        print("=" * 100)
        print(f"[Task {task_id}] [Candidate] {candidate_index} | [Pair] {pair_index}")
        print(f"INPUT  (H={len(x)}, W={len(x[0])})")
        print(grid_to_ascii(x))
        print("-" * 60)
        print(f"TARGET (H={len(y)}, W={len(y[0])})")
        print(grid_to_ascii(y))
        print("-" * 60)
        print("RAW MODEL RESPONSE (trimmed 2k):")
        print((raw or "")[:2000])
        print("-" * 60)
        if pred is None:
            print(f"PRED: <parse error>   note={note}")
        else:
            print(f"PRED   (H={len(pred)}, W={len(pred[0])})")
            print(grid_to_ascii(pred))
        print(f"CELL-ACCURACY: {acc:.1f}%")

    return {
        "pair_index": pair_index,
        "input": x,
        "target": y,
        "pred": pred,
        "note": note,
        "cell_accuracy": acc,
        "raw_text": (raw or "")[:2000],
    }

async def run_one_candidate(
    client: AsyncCompatClient,
    args: argparse.Namespace,
    task_id: str,
    candidate_index: int,
    instructions: str,
    pairs: List[Tuple[List[List[int]], List[List[int]]]],
    writer: RowWriter,
    print_lock: asyncio.Lock,
    global_sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    sem_pairs = asyncio.Semaphore(max(1, args.pair_concurrency))
    rows: List[Dict[str, Any]] = [None] * len(pairs)  # preserve indices

    async def _bounded_eval(pi: int, x: List[List[int]], y: List[List[int]]):
        async with sem_pairs:
            row = await eval_one_pair(client, args, task_id, candidate_index, pi, x, y, instructions, print_lock, global_sem)
            rows[pi] = row
            await writer.queue.put({"candidate_index": candidate_index, "row": row})

    tasks = [asyncio.create_task(_bounded_eval(i, x, y)) for i, (x, y) in enumerate(pairs)]
    await asyncio.gather(*tasks)

    # summary
    parsed_ok = sum(1 for r in rows if r and r["pred"] is not None)
    mean_acc = (sum(r["cell_accuracy"] for r in rows if r) / len(rows)) if rows else 0.0
    all_100 = all((r and r["cell_accuracy"] == 100.0) for r in rows)

    async with print_lock:
        print(f"[Task {task_id}] [candidate {candidate_index}] parsed_ok={parsed_ok}/{len(rows)}  mean_cell_acc={mean_acc:.2f}%  all_pairs_100={all_100}")

    return {
        "candidate_index": candidate_index,
        "parsed_ok_pairs": parsed_ok,
        "total_pairs": len(rows),
        "mean_cell_accuracy": round(mean_acc, 4),
        "all_pairs_100": all_100,
        "note": "",
    }

# ---------- Per-task driver ----------

async def run_one_task(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    task_id: str,
    global_sem: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Load pairs
    pairs = load_arc_training_pairs(args.root, args.split, task_id)

    # Load candidates for this task
    candidates_file = Path(str(args.candidates_file)).with_name(f"{task_id}.json") if args.candidates_layout_per_task else args.candidates_file
    with candidates_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    candidates = _normalize_candidates(data)
    if not candidates:
        async with print_lock:
            print(f"[Task {task_id}] No candidates found in {candidates_file}. Skipping.")
        return {"task_id": task_id, "num_pairs": len(pairs), "num_candidates": 0, "best_mean_acc": 0.0, "best_candidate_index": -1, "solved": False, "wall_time_sec": round(time.perf_counter() - t0, 3)}

    # Output files (per task)
    base_out = Path(args.out_root) if args.out_root else Path.cwd()
    out_dir = base_out / f"{task_id}_allc_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_jsonl = out_dir / f"{task_id}.per_pair.jsonl"
    per_pair_csv   = out_dir / f"{task_id}.per_pair.csv"
    summary_csv    = out_dir / f"{task_id}.candidates_summary.csv"

    async with print_lock:
        print("\n" + "#" * 100)
        print(f"[Task {task_id}] Starting… pairs={len(pairs)}  candidates={len(candidates)}  out={out_dir}")
        print("#" * 100)

    client = AsyncCompatClient(
        session=session,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.retries,
        backoff=args.backoff,
        use_guided_json=args.guided_json,
        use_guided_regex=args.guided_regex,
        use_response_format=(not args.no_response_format),
    )

    candidate_summaries: List[Dict[str, Any]] = []

    async with RowWriter(per_pair_jsonl, per_pair_csv) as writer:

        sem_candidates = asyncio.Semaphore(max(1, args.candidate_concurrency))

        async def _run_ci(ci: int, cand: Dict[str, Any]):
            async with sem_candidates:
                text = _extract_text_from_candidate(cand, args.instructions_key)
                if not text:
                    async with print_lock:
                        print(f"[Task {task_id}] [candidate {ci}] Skipped (no instruction text; try --instructions-key or --dump-candidates).")
                    return {
                        "candidate_index": ci,
                        "parsed_ok_pairs": 0,
                        "total_pairs": len(pairs),
                        "mean_cell_accuracy": 0.0,
                        "all_pairs_100": False,
                        "note": "no instruction text",
                    }
                async with print_lock:
                    print("\n" + "#" * 100)
                    preview = text[:240].replace("\n", " ")
                    print(f"[Task {task_id}] [candidate {ci}] INSTRUCTIONS (first 240 chars): {preview}")
                    print("#" * 100)
                return await run_one_candidate(client, args, task_id, ci, text, pairs, writer, print_lock, global_sem)

        tasks = [asyncio.create_task(_run_ci(ci, cand)) for ci, cand in enumerate(candidates)]
        results = await asyncio.gather(*tasks)
        candidate_summaries.extend(results)

    # Write per-task summary CSV
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_index", "parsed_ok_pairs", "total_pairs", "mean_cell_accuracy", "all_pairs_100", "note"])
        for s in candidate_summaries:
            w.writerow([
                s["candidate_index"],
                s["parsed_ok_pairs"],
                s["total_pairs"],
                s["mean_cell_accuracy"],
                s.get("all_pairs_100", False),
                s["note"],
            ])

    # Compute task solved/best stats
    best = max(candidate_summaries, key=lambda s: s["mean_cell_accuracy"]) if candidate_summaries else None
    best_mean = best["mean_cell_accuracy"] if best else 0.0
    best_idx = best["candidate_index"] if best else -1
    task_solved = any(s.get("all_pairs_100", False) for s in candidate_summaries)
    wall = round(time.perf_counter() - t0, 3)

    async with print_lock:
        status = "SOLVED ✅" if task_solved else "UNSOLVED ❌"
        print("\n" + "-" * 100)
        print(f"[Task {task_id}] SUMMARY: best_candidate={best_idx}  best_mean_acc={best_mean:.2f}%  parsed_ok/total={sum(s['parsed_ok_pairs'] for s in candidate_summaries)}/{len(pairs)*len(candidate_summaries)}  status={status}  wall_time={wall:.3f}s")
        print("-" * 100)
        print(f"[Task {task_id}] Wrote per-pair JSONL: {per_pair_jsonl}")
        print(f"[Task {task_id}] Wrote per-pair CSV   : {per_pair_csv}")
        print(f"[Task {task_id}] Wrote summary CSV    : {summary_csv}")

    return {
        "task_id": task_id,
        "num_pairs": len(pairs),
        "num_candidates": len(candidate_summaries),
        "best_mean_acc": round(best_mean, 4),
        "best_candidate_index": best_idx,
        "solved": task_solved,
        "wall_time_sec": wall,
    }

# ---------- Utility: collect task ids ----------

def _read_task_ids_from_file(p: Path) -> List[str]:
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def _infer_task_id_from_path(path: Path) -> Optional[str]:
    # expects filename like XXXXXXXX.json
    if path.suffix.lower() != ".json":
        return None
    return path.stem

def collect_task_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    # explicit list
    if args.task_ids:
        ids.extend(args.task_ids)
    # file with one id per line
    if args.task_file:
        ids.extend(_read_task_ids_from_file(args.task_file))
    # glob of candidate jsons (use their basenames as ids)
    if args.tasks_glob:
        for p in _glob.glob(args.tasks_glob):
            tid = _infer_task_id_from_path(Path(p))
            if tid:
                ids.append(tid)
    # de-dup, preserve order
    seen = set()
    out: List[str] = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    if args.task_id and args.task_id not in seen:
        out.append(args.task_id)
    if not out:
        raise SystemExit("No task ids provided. Use --task-id, --task-ids, --task-file, or --tasks-glob.")
    return out

# ---------- Multi-task driver ----------

async def run_many_tasks(args: argparse.Namespace) -> None:
    overall_t0 = time.perf_counter()
    task_ids = collect_task_ids(args)
    print(f"Discovered {len(task_ids)} task(s). Task-concurrency={args.task_concurrency} | candidate-concurrency={args.candidate_concurrency} | pair-concurrency={args.pair_concurrency} | global-requests={args.global_request_concurrency}")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)  # our semaphores control concurrency
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        print_lock = asyncio.Lock()
        global_sem = asyncio.Semaphore(max(1, args.global_request_concurrency))
        sem_tasks = asyncio.Semaphore(max(1, args.task_concurrency))

        per_task_results: List[Dict[str, Any]] = []

        async def _run_tid(tid: str):
            async with sem_tasks:
                try:
                    res = await run_one_task(session, args, tid, global_sem, print_lock)
                except Exception as e:
                    res = {"task_id": tid, "num_pairs": 0, "num_candidates": 0, "best_mean_acc": 0.0, "best_candidate_index": -1, "solved": False, "wall_time_sec": 0.0}
                    async with print_lock:
                        print(f"[Task {tid}] ERROR: {e}")
                per_task_results.append(res)

        await asyncio.gather(*[asyncio.create_task(_run_tid(tid)) for tid in task_ids])

    # Final multi-task report
    total_wall = round(time.perf_counter() - overall_t0, 3)
    per_task_results.sort(key=lambda r: r["task_id"])
    solved_ids = [r["task_id"] for r in per_task_results if r["solved"]]
    solved_count = len(solved_ids)

    print("\n" + "=" * 100)
    print("FINAL REPORT")
    print("-" * 100)
    print(f"Tasks solved: {solved_count}/{len(per_task_results)}")
    if solved_ids:
        print("Solved task_ids:", ", ".join(solved_ids))
    else:
        print("Solved task_ids: (none)")
    print("\nPer-task best mean accuracy:")
    for r in per_task_results:
        status = "SOLVED" if r["solved"] else "UNSOLVED"
        print(f"  {r['task_id']}: best_mean_acc={r['best_mean_acc']:.2f}%  best_candidate={r['best_candidate_index']}  status={status}  wall_time={r['wall_time_sec']:.3f}s")
    print("-" * 100)
    print(f"Overall wall-time: {total_wall:.3f}s")
    print("=" * 100)

    # Write tasks_summary.csv
    out_root = Path(args.out_root) if args.out_root else Path.cwd()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "tasks_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "num_pairs", "num_candidates", "best_candidate_index", "best_mean_acc", "solved", "wall_time_sec"])
        for r in per_task_results:
            w.writerow([r["task_id"], r["num_pairs"], r["num_candidates"], r["best_candidate_index"], r["best_mean_acc"], int(r["solved"]), r["wall_time_sec"]])
    print(f"Wrote tasks_summary.csv: {summary_path}")

# ---------- CLI ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ARC: concurrent all-candidates × all-pairs scorer across MANY tasks (with SOLVED detection and wall-time summaries).")
    # Data roots
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")

    # Single task (optional) and multi-task inputs
    p.add_argument("--task-id", type=str, help="Single task id (e.g. 00576224)")
    p.add_argument("--task-ids", type=str, nargs="*", help="Multiple task ids (space-separated)")
    p.add_argument("--task-file", type=Path, help="Text file with one task id per line")
    p.add_argument("--tasks-glob", type=str, help="Glob matching candidate JSONs (e.g. '/content/outputs/training/*.json'). Basenames become task_ids")

    # Candidates
    p.add_argument("--candidates-file", type=Path, required=True, help="Path to candidates JSON. If --candidates-layout-per-task, filename is replaced by <task_id>.json in the same folder.")
    p.add_argument("--candidates-layout-per-task", action="store_true", help="Use <candidates_file_dir>/<task_id>.json for each task")
    p.add_argument("--instructions-key", type=str, default="", help="Dot-path to instruction text if nested")
    p.add_argument("--dump-candidates", action="store_true", help="Preview text-like fields (no generation) for the (single) --task-id")
    p.add_argument("--dump-limit", type=int, default=10)

    # Model + runtime
    p.add_argument("--model", type=str, required=True, help="Model name")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--backoff", type=float, default=0.6)

    # Output constraints
    p.add_argument("--guided-json", action="store_true", help="Use vLLM guided_json (best)")
    p.add_argument("--guided-regex", action="store_true", help="Use vLLM guided_regex (fallback)")
    p.add_argument("--no-response-format", action="store_true", help="Do NOT send response_format hint")
    p.add_argument("--enforce-shape", action="store_true", help="Require output to match target HxW")

    # Concurrency knobs
    p.add_argument("--pair-concurrency", type=int, default=8, help="Parallel pairs per candidate (per task)")
    p.add_argument("--candidate-concurrency", type=int, default=1, help="Parallel candidates per task")
    p.add_argument("--task-concurrency", type=int, default=2, help="Parallel tasks")
    p.add_argument("--global-request-concurrency", type=int, default=16, help="Global limit on in-flight HTTP requests across ALL tasks")

    # Output root
    p.add_argument("--out-root", type=Path, help="Base directory (default: current working dir). Each task writes to <out_root>/<task_id>_allc_out")
    return p

def main():
    args = build_arg_parser().parse_args()

    # Optional: dump mode (single task)
    if args.dump_candidates:
        if not args.task_id:
            raise SystemExit("--dump-candidates requires --task-id (and a matching candidates file)")
        dump_candidates_only(args.candidates_file, limit=args.dump_limit)
        return

    asyncio.run(run_many_tasks(args))

if __name__ == "__main__":
    main()


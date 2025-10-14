#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-AGI Instruction Generator (multi-candidate, instruction-only)
Async + Parallel for OpenAI-compatible servers (vLLM / TGI)
Tuned for Colab L4; robust context budgeting to avoid 400s.

Environment:
  OPENAI_BASE_URL  (e.g., http://127.0.0.1:8000/v1)
  OPENAI_API_KEY   (any non-empty string; header required)

Outputs:
  - ./outputs/<split>/<task_id>.json            (bundle of instruction candidates)
  - ./outputs/<split>/nl_descriptions.jsonl     (one line per candidate)

Usage example:
  python arc_instructions_generate.py \
    --root /content/ARC-AGI-2/data \
    --split training \
    --max-tasks 500 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-candidates 6 \
    --concurrency 4 \
    --ascii-style \
    --ctx-limit 32768 --ctx-margin 512 --max-tokens 256 \
    --outputs /content/outputs
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import aiohttp

# -------------------------
# ARC helpers
# -------------------------

def load_arc_task(task_path: Path) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_tasks(root: Path, split: str) -> List[Path]:
    pattern = str(root / split / "*.json")
    paths = [Path(p) for p in glob.glob(pattern)]
    paths.sort()
    return paths

def grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)

def grid_to_ascii(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def grid_to_rle(grid: List[List[int]]) -> str:
    """Run-length encode each row to reduce tokens while preserving structure."""
    out_lines: List[str] = []
    for row in grid:
        if not row:
            out_lines.append("")
            continue
        parts: List[str] = []
        cur = row[0]; cnt = 1
        for v in row[1:]:
            if v == cur:
                cnt += 1
            else:
                parts.append(f"{cur}x{cnt}")
                cur = v; cnt = 1
        parts.append(f"{cur}x{cnt}")
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)

# -------------------------
# Jeremy-Berman-style prompt
# -------------------------

BERMAN_SYSTEM = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n"
    "Find the common pattern that transforms each input grid into its corresponding output grid, "
    "based on the training examples below.\n\n"
    "Your task is to write clear instructions that describe this transformation pattern. These instructions must:\n"
    "- Apply consistently to ALL training examples (the same rule works for every input→output pair).\n"
    "- Be general enough to work on new test cases.\n"
    "- Be intuitive and easy to understand.\n"
    "- Describe the pattern without referencing specific example numbers or positions.\n\n"
    "The transformation pattern should be simple and logical — these puzzles are designed to have elegant, "
    "intuitive solutions that humans can readily grasp.\n\n"
    "Write your instructions as a clear, step-by-step process that someone could follow "
    "to transform any input grid into the correct output grid.\n"
)

BERMAN_OUTPUT_GUIDE = (
    "Return ONLY:\n"
    "1) A one-sentence rule summary.\n"
    "2) A numbered, step-by-step procedure.\n"
    "3) Brief notes for edge cases/ambiguities."
)

def format_examples_ascii(task: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Here are the training examples:")
    for i, pair in enumerate(task.get("train", [])):
        inp = pair["input"]; out = pair["output"]
        ih, iw = grid_shape(inp); oh, ow = grid_shape(out)
        lines.append(f"Training Example {i+1}")
        lines.append(f"Input (H={ih}, W={iw})")
        lines.append(grid_to_ascii(inp))
        lines.append(f"Output (H={oh}, W={ow})")
        lines.append(grid_to_ascii(out))
        lines.append("")
    lines.append("-- End of Training Examples --")
    return "\n".join(lines).strip()

def format_examples_compact(task: Dict[str, Any]) -> str:
    """Compact minified JSON version to save tokens."""
    chunks: List[str] = []
    for i, pair in enumerate(task.get("train", [])):
        inp = pair["input"]; out = pair["output"]
        ih, iw = grid_shape(inp); oh, ow = grid_shape(out)
        chunks.append(f"ex{i+1}: in({ih}x{iw})→out({oh}x{ow})")
        chunks.append(json.dumps({"input": inp, "output": out}, separators=(",", ":")))
    return "\n".join(chunks).strip()

def format_examples_rle(task: Dict[str, Any]) -> str:
    lines: List[str] = []
    for i, pair in enumerate(task.get("train", [])):
        inp = pair["input"]; out = pair["output"]
        ih, iw = grid_shape(inp); oh, ow = grid_shape(out)
        lines.append(f"Training Example {i+1} (in {ih}x{iw} -> out {oh}x{ow})")
        lines.append("Input (RLE)")
        lines.append(grid_to_rle(inp))
        lines.append("Output (RLE)")
        lines.append(grid_to_rle(out))
        lines.append("")
    lines.append("-- End of Training Examples --")
    return "\n".join(lines).strip()

def build_instruction_prompt(task_id: str, task: Dict[str, Any], *, style: str) -> str:
    builders: Dict[str, Callable[[Dict[str, Any]], str]] = {
        "ascii": format_examples_ascii,
        "compact": format_examples_compact,
        "rle": format_examples_rle,
    }
    examples = builders.get(style, format_examples_compact)(task)
    return f"{BERMAN_SYSTEM}\n\nTask: {task_id}\n\n{examples}\n\n{BERMAN_OUTPUT_GUIDE}"

# -------------------------
# Token budgeting (avoid HTTP 400)
# -------------------------

def approx_token_count(text: str) -> int:
    """Conservative token estimator: ~3.2 chars per token; add 10% safety."""
    return int(max(1, ((len(text) / 3.2) * 1.1)))

def budget_max_tokens(prompt_text: str, *, ctx_limit: int, ctx_margin: int, min_completion: int, hard_cap: int) -> int:
    prompt_tokens = approx_token_count(prompt_text)
    allowance = ctx_limit - prompt_tokens - ctx_margin
    return max(min_completion, min(hard_cap, max(1, allowance)))

# -------------------------
# Async OpenAI-compatible client
# -------------------------

@dataclass
class GenResult:
    text: Optional[str]
    error: Optional[str]

class AsyncCompatClient:
    """Minimal async client for OpenAI-compatible endpoints (/v1/completions by default)."""
    def __init__(
        self,
        timeout: int = 180,
        max_tokens: int = 384,
        retries: int = 5,
        backoff: float = 0.6,
        use_chat: bool = False,
    ) -> None:
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.use_chat = use_chat
        self.base_url = f"{base}/chat/completions" if use_chat else f"{base}/completions"
        self.api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.default_max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
            self._session = None

    def _payload(self, *, model: str, prompt: str, temperature: float, seed: Optional[int], max_tokens: Optional[int]) -> Dict[str, Any]:
        max_toks = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        if self.use_chat:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(max(0.0, min(2.0, temperature))),
                "max_tokens": max_toks,
                "n": 1,
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": float(max(0.0, min(2.0, temperature))),
                "max_tokens": max_toks,
                "n": 1,
            }
        if seed is not None:
            payload["seed"] = int(seed)
        return payload

    async def generate(self, *, model: str, prompt: str, temperature: float = 0.2, seed: Optional[int] = None, max_tokens: Optional[int] = None) -> GenResult:
        assert self._session is not None, "AsyncCompatClient must be used with 'async with'"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = self._payload(model=model, prompt=prompt, temperature=temperature, seed=seed, max_tokens=max_tokens)

        last_err: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                async with self._session.post(self.base_url, headers=headers, json=payload) as resp:
                    text_body = await resp.text()
                    if resp.status != 200:
                        last_err = f"HTTP {resp.status}: {text_body[:400]}"
                    else:
                        try:
                            data = json.loads(text_body)
                        except Exception as je:
                            last_err = f"JSON decode error: {je} | body[:200] = {text_body[:200]}"
                            data = None
                        if data is not None:
                            choices = data.get("choices", [])
                            if choices:
                                if self.use_chat:
                                    msg = choices[0].get("message", {})
                                    content = msg.get("content", "")
                                    return GenResult(text=content, error=None)
                                else:
                                    return GenResult(text=choices[0].get("text", ""), error=None)
                            last_err = "no choices returned"
            except Exception as e:
                last_err = f"request error: {str(e)}"
            await asyncio.sleep(self.backoff * (attempt + 1))
        return GenResult(text=None, error=last_err)

# -------------------------
# Output writers
# -------------------------

def ensure_out_dirs(base: Path, split: str) -> Tuple[Path, Path]:
    split_dir = base / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "nl_descriptions.jsonl").touch(exist_ok=True)
    return split_dir, split_dir / "nl_descriptions.jsonl"

def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_per_task_json(out_dir: Path, task_id: str, payload: Dict[str, Any]) -> None:
    with open(out_dir / f"{task_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -------------------------
# Async runner (generation only)
# -------------------------

@dataclass
class CandRec:
    candidate_id: int
    ts: str
    prompt: str
    nl_description: str
    model: str
    temperature: float
    seed: int
    style: str

async def process_task(*, client: AsyncCompatClient, args: argparse.Namespace, task_path: Path, idx: int, total: int) -> None:
    task_id = task_path.stem
    out_dir, jsonl_path = ensure_out_dirs(args.outputs, args.split)
    per_task_path = out_dir / f"{task_id}.json"

    if per_task_path.exists() and not args.overwrite:
        print(f"[{idx}/{total}] {task_id}: exists, skip (use --overwrite to regenerate)")
        return

    try:
        task = load_arc_task(task_path)
    except Exception as e:
        print(f"[{idx}/{total}] {task_id}: ERROR loading: {e}")
        return

    print(f"[{idx}/{total}] {task_id}: target candidates={args.num_candidates}")

    # Style preference & fallback order
    style_order: List[str] = []
    if args.ascii_style:
        style_order.append("ascii")
    if args.compact:
        style_order.append("compact")
    style_order += ["rle", "compact", "ascii"]
    seen = set()
    style_order = [s for s in style_order if not (s in seen or seen.add(s))]

    def build_prompt_with_budget(style: str) -> Tuple[str, int]:
        prompt = build_instruction_prompt(task_id, task, style=style)
        budget = budget_max_tokens(
            prompt,
            ctx_limit=args.ctx_limit,
            ctx_margin=args.ctx_margin,
            min_completion=args.min_completion,
            hard_cap=args.max_tokens,
        )
        return prompt, budget

    # Pick a style that fits budget
    selected = None
    for st in style_order:
        p_try, b_try = build_prompt_with_budget(st)
        est = approx_token_count(p_try) + args.min_completion + args.ctx_margin
        if est <= args.ctx_limit:
            selected = (st, p_try, b_try)
            break
    if selected is None:
        st, p_try, b_try = "compact", *build_prompt_with_budget("compact")
        selected = (st, p_try, b_try)

    base_style, base_prompt, base_budget = selected
    print(f"      [prompt-style] {base_style} | est_prompt_tokens≈{approx_token_count(base_prompt)} budget={base_budget}")

    async def _gen_slot(slot_k: int) -> Optional[CandRec]:
        for tnum in range(1, args.candidate_tries + 1):
            seed_k = (args.seed or 0) + (slot_k * 13 + tnum)
            temp_k = max(0.0, min(1.0, args.temperature + 0.05 * ((tnum % 3) - 1)))
            curr_prompt = base_prompt
            curr_budget = base_budget
            curr_style = base_style

            res = await client.generate(model=args.model, prompt=curr_prompt, temperature=temp_k, seed=seed_k, max_tokens=curr_budget)
            if res.text is None and res.error and ("maximum context length" in res.error or "context length" in res.error or "HTTP 400" in res.error):
                # Try other styles or shrink budget
                for fs in ["compact", "rle", "ascii"]:
                    if fs == curr_style:
                        continue
                    new_p, new_b = build_prompt_with_budget(fs)
                    new_est = approx_token_count(new_p) + args.min_completion + args.ctx_margin
                    if new_est <= args.ctx_limit:
                        curr_prompt, curr_budget, curr_style = new_p, new_b, fs
                        print(f"      [auto-downgrade] switched to {fs} | est_tokens≈{approx_token_count(new_p)} budget={new_b}")
                        break
                else:
                    curr_budget = max(args.min_completion, int(curr_budget * 0.6))
                res = await client.generate(model=args.model, prompt=curr_prompt, temperature=temp_k, seed=seed_k, max_tokens=curr_budget)

            if res.text is None:
                print(f"  - cand {slot_k} try {tnum}/{args.candidate_tries}: error: {res.error or 'unknown'}")
                continue

            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec = CandRec(
                candidate_id=slot_k,
                ts=ts,
                prompt=curr_prompt,
                nl_description=res.text,
                model=args.model,
                temperature=temp_k,
                seed=seed_k,
                style=curr_style,
            )
            write_jsonl(jsonl_path, {
                "ts": ts,
                "task_id": task_id,
                "split": args.split,
                "candidate_id": slot_k,
                "model": args.model,
                "temperature": temp_k,
                "seed": seed_k,
                "nl_description": res.text,
                "error": None,
                "style": curr_style,
            })
            print(f"    ✓ cand {slot_k}: GENERATED ({curr_style})")
            return rec
        print(f"  - cand {slot_k}: FAILED after {args.candidate_tries} tries — leaving this slot empty")
        return None

    sem_slots = asyncio.Semaphore(args.concurrency)

    async def _slot_driver(k: int) -> Optional[CandRec]:
        async with sem_slots:
            return await _gen_slot(k)

    tasks = [asyncio.create_task(_slot_driver(k)) for k in range(args.num_candidates)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    accepted = [r.__dict__ for r in results if r is not None]

    bundle = {
        "task_id": task_id,
        "split": args.split,
        "num_candidates_requested": args.num_candidates,
        "num_candidates_generated": len(accepted),
        "candidates": accepted,
        "note": "Instruction-only generation; robust anti-400 budgeting.",
    }
    save_per_task_json(out_dir, task_id, bundle)
    print(f"[{idx}/{total}] {task_id}: wrote {len(accepted)} instruction candidates")

async def amain(args: argparse.Namespace) -> None:
    random.seed(args.seed or None)

    tasks = list_tasks(args.root, args.split)
    if args.task_id:
        tasks = [p for p in tasks if p.stem == args.task_id]
        if not tasks:
            print(f"No task found matching --task-id {args.task_id}", file=sys.stderr)
            sys.exit(2)

    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        print("No tasks found. Check --root and --split.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(tasks)} tasks. Generating up to {args.num_candidates} instruction candidates per task…", flush=True)

    async with AsyncCompatClient(timeout=args.timeout, max_tokens=args.max_tokens, retries=args.retries, backoff=args.backoff, use_chat=args.use_chat) as client:
        for idx, task_path in enumerate(tasks, 1):
            await process_task(client=client, args=args, task_path=task_path, idx=idx, total=len(tasks))
    print("Done.")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ARC-AGI instruction generator — Async & Parallel (OpenAI-compatible)")
    p.add_argument("--root", type=Path, required=True, help="Path to ARC-AGI dataset root (contains training/ evaluation/)")
    p.add_argument("--split", type=str, choices=["training", "evaluation"], default="training")
    p.add_argument("--max-tasks", type=int, default=0, help="Optional cap on number of tasks (0 = all)")

    # Model & temps
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model HF id served by your backend")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature for describer")

    # Concurrency
    p.add_argument("--concurrency", type=int, default=4, help="Max concurrent candidate slots")

    # Client runtime
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=180, help="Request timeout seconds")
    p.add_argument("--retries", type=int, default=5, help="HTTP retry attempts per request")
    p.add_argument("--backoff", type=float, default=0.6, help="Backoff seconds multiplier between retries")
    p.add_argument("--max-tokens", type=int, default=320, help="Global cap; per-request budgeting may use lower")
    p.add_argument("--use-chat", action="store_true", help="Use /v1/chat/completions instead of /v1/completions")

    # Budgeting (to prevent context 400s)
    p.add_argument("--ctx-limit", type=int, default=32768, help="Server context window in tokens")
    p.add_argument("--ctx-margin", type=int, default=512, help="Safety tokens reserved for server/system overhead")
    p.add_argument("--min-completion", type=int, default=96, help="Minimum completion budget after prompt budgeting")

    # Prompt style prefs
    p.add_argument("--compact", action="store_true", help="Prefer compact JSON examples")
    p.add_argument("--ascii-style", action="store_true", help="Prefer ASCII examples (may be auto-downgraded if too large)")

    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if per-task JSON exists")
    p.add_argument("--num-candidates", type=int, default=6, help="Number of NL candidates per task")
    p.add_argument("--candidate-tries", type=int, default=4, help="Regenerations per candidate slot until one succeeds")
    p.add_argument("--task-id", type=str, default="", help="Only run this task id (e.g. 00576224)")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("Interrupted.")

if __name__ == "__main__":
    main()

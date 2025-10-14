#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthesize Python programs + DSL operations for ARC tasks using scored instruction sets.

Pipeline per task:
  1) Read scoring summary (candidates_summary.csv) and select best instruction candidate(s).
  2) Load the task training pairs and the chosen instruction text(s).
  3) Prompt model to emit JSON:
        {
          "python_code": "... def transform(grid): ...",
          "dsl_operations": [{"name","signature","description"},...],
          "rationale": "..."  # optional
        }
     (Uses guided JSON schema or response_format schema; streaming is optional and cleaned up.)
  4) Print DSL + code immediately after parsing JSON.
  5) Validate the produced Python by exec'ing and running transform() on training inputs with a watchdog timeout.
  6) Write program.py, dsl_operations.json, synthesis.json; print telemetry.

Concurrency:
  • --task-concurrency: number of tasks synthesized in parallel
  • --global-request-concurrency: total HTTP requests across all tasks

Tip: run with unbuffered IO for snappier logs:
  PYTHONUNBUFFERED=1 python -u arc_synthesize_from_scores.py ...
"""

from __future__ import annotations

import argparse
import asyncio
import aiohttp
import csv
import glob
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
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

def load_arc_pairs(root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    f = root / split / f"{task_id}.json"
    with f.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = data.get("train") or data.get("training") or data.get("pairs")
    if not pairs or not isinstance(pairs, list):
        raise ValueError(f"No training pairs in {f}")
    norm = []
    for p in pairs:
        x, y = p.get("input"), p.get("output")
        if not (is_valid_grid(x) and is_valid_grid(y)):
            raise ValueError(f"Invalid grid(s) in {f}")
        norm.append({"input": x, "output": y})
    return norm

def read_candidates(candidates_root: Path, split: str, task_id: str) -> List[Dict[str, Any]]:
    cpath = candidates_root / split / f"{task_id}.json"
    with cpath.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cands = data.get("candidates")
    if not isinstance(cands, list):
        raise ValueError(f"No candidates list in {cpath}")
    return cands

def extract_instruction_text(cand: Dict[str, Any]) -> Optional[str]:
    for k in ("instructions", "instruction", "text", "content", "nl", "prompt", "instruction_text"):
        v = cand.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("bullets", "lines", "steps", "instructions_list"):
        v = cand.get(k)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            joined = "\n".join(x.strip() for x in v if x.strip())
            if joined:
                return joined
    msgs = cand.get("messages")
    if isinstance(msgs, list) and msgs:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
    for k in ("initial", "revised", "best", "final", "candidate", "draft"):
        sub = cand.get(k)
        if isinstance(sub, dict):
            for kk in ("instructions", "text", "content", "nl"):
                v = sub.get(kk)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for kk in ("bullets", "lines", "steps"):
                v = sub.get(kk)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    joined = "\n".join(x.strip() for x in v if x.strip())
                    if joined:
                        return joined
    return None

# ---------------- Score selection ----------------

@dataclass
class CandidateScore:
    candidate_index: int
    parsed_ok_pairs: int
    total_pairs: int
    mean_cell_accuracy: float

def parse_summary_csv(path: Path) -> List[CandidateScore]:
    out: List[CandidateScore] = []
    with path.open("r", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                out.append(
                    CandidateScore(
                        candidate_index=int(row["candidate_index"]),
                        parsed_ok_pairs=int(row["parsed_ok_pairs"]),
                        total_pairs=int(row["total_pairs"]),
                        mean_cell_accuracy=float(row["mean_cell_accuracy"]),
                    )
                )
            except Exception:
                continue
    return out

def pick_candidates_for_task(scores: List[CandidateScore], policy: str, topk: int) -> List[int]:
    solved = [s for s in scores if s.parsed_ok_pairs == s.total_pairs and abs(s.mean_cell_accuracy - 100.0) < 1e-6]
    if policy == "solved_only":
        chosen = sorted(solved, key=lambda s: s.candidate_index)[:topk]
        return [s.candidate_index for s in chosen]
    best = sorted(scores, key=lambda s: (s.mean_cell_accuracy, s.parsed_ok_pairs), reverse=True)
    chosen = (sorted(solved, key=lambda s: s.candidate_index) + [s for s in best if s not in solved])[:topk]
    return [s.candidate_index for s in chosen]

# ---------------- Model client (with optional clean streaming) ----------------

@dataclass
class GenResult:
    text: Optional[str]
    error: Optional[str]

class ChatClient:
    def __init__(self, session: aiohttp.ClientSession, *, timeout: int, retries: int, backoff: float, live_prefix: str = ""):
        base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.key = os.environ.get("OPENAI_API_KEY", "sk")
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.backoff = backoff
        self.live_prefix = live_prefix

    def _schema(self) -> Dict[str, Any]:
        return {
            "name": "arc_program_synthesis",
            "schema": {
                "type": "object",
                "properties": {
                    "python_code": {"type": "string"},
                    "dsl_operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "signature": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "signature", "description"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                    "rationale": {"type": "string"}
                },
                "required": ["python_code", "dsl_operations"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    async def generate(
        self, *,
        model: str, system: str, user: str, max_tokens: int,
        use_guided_json: bool, use_response_format: bool,
        global_sem: asyncio.Semaphore, stream: bool = False, stream_print: bool = False
    ) -> GenResult:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        schema = self._schema()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if use_guided_json:
            payload["guided_json"] = schema["schema"]
        elif use_response_format:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        if stream:
            payload["stream"] = True

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                async with global_sem:
                    async with self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout) as resp:
                        if not stream:
                            body = await resp.text()
                            if resp.status != 200:
                                last_err = f"HTTP {resp.status}: {body[:400]}"
                            else:
                                try:
                                    data = json.loads(body)
                                    ch = data.get("choices", [])
                                    if ch:
                                        return GenResult(text=ch[0].get("message", {}).get("content", ""), error=None)
                                    last_err = "no choices"
                                except Exception as e:
                                    last_err = f"decode content error: {e} | content[:300]={body[:300]}"
                        else:
                            # Clean streaming print: buffer until newline or top-level JSON closes
                            buf = []
                            brace_level = 0
                            started_json = False
                            printed_intro = False
                            async for line_bytes in resp.content:
                                line = line_bytes.decode("utf-8", errors="ignore").rstrip()
                                if not line or not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                except Exception:
                                    continue
                                delta = (((data.get("choices") or [{}])[0]).get("delta") or {})
                                chunk = delta.get("content")
                                if not chunk:
                                    continue
                                buf.append(chunk)
                                if stream_print:
                                    if not printed_intro:
                                        sys.stdout.write(self.live_prefix + "(streaming JSON...)\n"); sys.stdout.flush()
                                        printed_intro = True
                                    for ch in chunk:
                                        if ch == "{":
                                            brace_level += 1
                                            started_json = True
                                        elif ch == "}":
                                            brace_level -= 1
                                    if ("\n" in chunk) or (started_json and brace_level == 0):
                                        sys.stdout.write(self.live_prefix + "".join(buf) + "\n"); sys.stdout.flush()
                                        buf.clear()
                            result = "".join(buf)
                            if stream_print and result:
                                sys.stdout.write(self.live_prefix + result + "\n"); sys.stdout.flush()
                            return GenResult(text=result, error=None)
            except Exception as e:
                last_err = f"request error: {e}"
            if attempt < self.retries:
                sys.stdout.write(f"[net] retry {attempt+1}/{self.retries}  last_err={last_err}\n"); sys.stdout.flush()
            await asyncio.sleep(self.backoff * (attempt + 1))
        return GenResult(text=None, error=last_err)

# ---------------- Prompting ----------------

SYSTEM = (
    "You are an expert ARC program synthesizer. "
    "Given several high-quality instruction sets and the task's training pairs, "
    "produce a single deterministic Python function `transform(grid)` that implements the rule. "
    "Also propose a small list of DSL primitives (name, signature, description) that would express this solution."
)

def build_user_prompt(task_id: str,
                      pairs: List[Dict[str, Any]],
                      instruction_texts: List[str],
                      max_pairs: int = 6) -> str:
    subset = pairs[:max_pairs]
    lines = [
        f"TASK {task_id}: Synthesize a definitive Python implementation and a DSL op list.",
        "",
        "TRAINING PAIRS (INPUT -> OUTPUT):"
    ]
    for i, p in enumerate(subset):
        x = json.dumps(p["input"], separators=(",", ":"))
        y = json.dumps(p["output"], separators=(",", ":"))
        lines.append(f"- Pair {i}: INPUT={x}  OUTPUT={y}")

    lines.append("\nCANDIDATE INSTRUCTION SETS (most relevant first):")
    for k, instr in enumerate(instruction_texts, 1):
        snippet = instr.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " ..."
        lines.append(f"--- Candidate #{k} ---\n{snippet}")

    lines.append("""
REQUIREMENTS:
- Return STRICT JSON with fields:
  {
    "python_code": "<full code for a module that defines def transform(grid): ...>",
    "dsl_operations": [{"name": "...", "signature": "...", "description": "..."}, ...],
    "rationale": "<brief reasoning (optional)>"
  }
- The Python must be deterministic, pure, and only use the standard library. No prints.
- Accept and return rectangular List[List[int]] grids (0..9). Do NOT read files or network.
- If resizing is required by the rule, compute new shape rigorously.
- Prefer clear, small helpers (e.g., flood_fill, bbox, upsample/downsample, tile, reflect, rotate, draw_line, paint_mask).
- Avoid overfitting: the program must generalize to new inputs consistent with the rule.
""")
    return "\n".join(lines)

# ---------------- Code validation ----------------

def _safe_exec(code: str) -> Tuple[Optional[Any], Optional[str]]:
    g: Dict[str, Any] = {}
    try:
        exec(code, g, g)
    except Exception as e:
        return None, f"exec error: {e}\n{traceback.format_exc()}"
    fn = g.get("transform")
    if not callable(fn):
        return None, "no transform(grid) function exported"
    return fn, None

def _run_transform(fn, grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    try:
        out = fn(grid)
    except Exception as e:
        return None, f"runtime error: {e}\n{traceback.format_exc()}"
    if not is_valid_grid(out):
        return None, "transform returned invalid grid"
    return out, None

def validate_code_on_pairs(code: str, pairs: List[Dict[str, Any]], max_pairs: int = 8) -> Dict[str, Any]:
    fn, err = _safe_exec(code)
    if err:
        return {"ok": False, "error": err, "per_pair": []}
    results = []
    ok_all = True
    for i, p in enumerate(pairs[:max_pairs]):
        pred, err2 = _run_transform(fn, p["input"])
        if err2:
            results.append({"pair_index": i, "ok": False, "error": err2})
            ok_all = False
        else:
            same_shape = (len(pred) == len(p["output"]) and len(pred[0]) == len(p["output"][0]))
            same_cells = same_shape and all(
                pred[r][c] == p["output"][r][c] for r in range(len(pred)) for c in range(len(pred[0]))
            )
            results.append({"pair_index": i, "ok": same_cells, "same_shape": same_shape})
            ok_all = ok_all and same_cells
    return {"ok": ok_all, "per_pair": results}

# ---------------- Synthesis core ----------------

async def synthesize_one(
    client: ChatClient,
    args: argparse.Namespace,
    task_id: str,
    scores_file: Path,
    global_sem: asyncio.Semaphore,
    print_lock: asyncio.Lock,
) -> None:
    t0 = time.perf_counter()
    out_task_dir = Path(args.out_dir) / task_id
    out_task_dir.mkdir(parents=True, exist_ok=True)
    synth_json = out_task_dir / "synthesis.json"
    program_py  = out_task_dir / "program.py"
    dsl_json    = out_task_dir / "dsl_operations.json"

    async with print_lock:
        print(f"\n[synth:{task_id}] ▶ start   scores={scores_file}"); sys.stdout.flush()

    # 1) Read scores + pick candidates
    scores = parse_summary_csv(scores_file)
    if not scores:
        async with print_lock:
            print(f"[synth:{task_id}] no scores in {scores_file}"); sys.stdout.flush()
        return
    chosen_idx = pick_candidates_for_task(scores, policy=args.selection_policy, topk=args.max_instruction_sets)

    async with print_lock:
        best_line = ", ".join(f"{s.candidate_index}:{s.mean_cell_accuracy:.2f}%" for s in sorted(scores, key=lambda x: x.mean_cell_accuracy, reverse=True)[:3])
        print(f"[synth:{task_id}] selection policy={args.selection_policy}  chosen={chosen_idx}  top3={best_line}"); sys.stdout.flush()

    # 2) Load candidates + instruction text
    try:
        cands = read_candidates(Path(args.candidates_root), args.split, task_id)
    except Exception as e:
        async with print_lock:
            print(f"[synth:{task_id}] ERROR reading candidates: {e}"); sys.stdout.flush()
        return

    instrs: List[str] = []
    for i in chosen_idx:
        if 0 <= i < len(cands):
            t = extract_instruction_text(cands[i])
            if t:
                instrs.append(t)
    if not instrs:
        async with print_lock:
            print(f"[synth:{task_id}] selected indices have no instruction text"); sys.stdout.flush()
        return

    # 3) Load training pairs
    pairs = load_arc_pairs(Path(args.root), args.split, task_id)

    # 4) Build prompt
    user = build_user_prompt(task_id, pairs, instrs, max_pairs=args.max_pairs_in_prompt)
    async with print_lock:
        print(f"[synth:{task_id}] prompt built  pairs_in_prompt={min(len(pairs), args.max_pairs_in_prompt)}  instrs={len(instrs)}"); sys.stdout.flush()
        if args.print_prompt_preview:
            preview = user[:args.print_limit or 2000]
            print(f"[synth:{task_id}] PROMPT PREVIEW:\n{preview}\n--- END PROMPT PREVIEW ---"); sys.stdout.flush()

    # 5) Model call with streaming gate (disable streaming if guided_json enabled)
    stream_flag = bool(args.stream_model) and not bool(args.guided_json)
    async with print_lock:
        print(f"[synth:{task_id}] ▶ model call  model={args.model}  guided_json={bool(args.guided_json)}  stream={stream_flag}"); sys.stdout.flush()
    res = await client.generate(
        model=args.model,
        system=SYSTEM,
        user=user,
        max_tokens=args.max_tokens,
        use_guided_json=bool(args.guided_json),
        use_response_format=not args.no_response_format,
        global_sem=global_sem,
        stream=stream_flag,
        stream_print=(stream_flag and args.stream_live),
    )
    raw = (res.text or "").strip()
    async with print_lock:
        print(f"[synth:{task_id}] ◀ model done  chars={len(raw)}  error={res.error is not None}"); sys.stdout.flush()

    # 6) Parse JSON
    parsed: Dict[str, Any] = {}
    parse_err = None
    try:
        parsed = json.loads(raw)
    except Exception as e:
        parse_err = f"json parse error: {e}"

    code = (parsed.get("python_code") if isinstance(parsed, dict) else None) or ""
    dsl_ops = (parsed.get("dsl_operations") if isinstance(parsed, dict) else None) or []

    # ---- Print artifacts immediately (before validation) ----
    async with print_lock:
        print(f"[synth:{task_id}] parsed JSON: has_code={bool(code)} has_dsl={isinstance(dsl_ops, list)}"); sys.stdout.flush()
        if args.print_dsl and isinstance(dsl_ops, list):
            to_show = json.dumps(dsl_ops, ensure_ascii=False, indent=2)
            if args.print_limit and len(to_show) > args.print_limit:
                to_show = to_show[:args.print_limit] + " ... [truncated]"
            print(f"[synth:{task_id}] DSL OPERATIONS:\n{to_show}"); sys.stdout.flush()
        if args.print_code and code:
            to_show = code
            if args.print_limit and len(to_show) > args.print_limit:
                to_show = to_show[:args.print_limit] + "\n# ... [truncated]"
            print(f"[synth:{task_id}] PYTHON PROGRAM:\n{to_show}"); sys.stdout.flush()

    # 7) Validate code with watchdog timeout
    validation = {"ok": False, "per_pair": [], "error": "no code"}
    if code:
        async with print_lock:
            print(f"[synth:{task_id}] ▶ validate on {min(len(pairs), args.max_pairs_validate)} pairs (timeout={args.validation_timeout}s)"); sys.stdout.flush()
        async def _validate():
            return validate_code_on_pairs(code, pairs, max_pairs=min(len(pairs), args.max_pairs_validate))
        try:
            validation = await asyncio.wait_for(_validate(), timeout=args.validation_timeout)
        except asyncio.TimeoutError:
            validation = {"ok": False, "per_pair": [], "error": f"validation timeout after {args.validation_timeout}s"}

    # 8) Write artifacts
    meta = {
        "task_id": task_id,
        "scores_file": str(scores_file),
        "chosen_candidate_indices": chosen_idx,
        "guided_json": bool(args.guided_json),
        "selection_policy": args.selection_policy,
        "raw_model_text_len": len(raw),
        "parse_error": parse_err,
        "validation": validation,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
    }
    synth_payload = {
        "meta": meta,
        "model_json": parsed if isinstance(parsed, dict) else {"raw": raw},
    }
    synth_json.write_text(json.dumps(synth_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if code:
        program_py.write_text(code, encoding="utf-8")
    if isinstance(dsl_ops, list):
        dsl_json.write_text(json.dumps(dsl_ops, ensure_ascii=False, indent=2), encoding="utf-8")

    # 9) Done line
    async with print_lock:
        status = "OK" if validation.get("ok") else "FAIL"
        print(f"[synth:{task_id}] ▶ done    status={status}  pairs={len(pairs)}  wall={meta['wall_time_sec']:.2f}s"); sys.stdout.flush()
        if parse_err:
            print(f"[synth:{task_id}] note: {parse_err}"); sys.stdout.flush()

# ---------------- Driver ----------------

def find_summary_files(scores_glob: str) -> List[Tuple[str, Path]]:
    out = []
    for p in glob.glob(scores_glob, recursive=True):
        path = Path(p)
        if path.name.endswith("candidates_summary.csv"):
            tid = path.stem.replace(".candidates_summary", "")
            if tid.endswith(".csv"):
                tid = tid[:-4]
            parent = path.parent.name
            if parent.endswith("_allc_out"):
                maybe = parent[:-9]
                if maybe:
                    tid = maybe
            out.append((tid, path))
    # de-dup by task_id, keep first occurrence
    seen = set()
    uniq = []
    for tid, p in out:
        if tid not in seen:
            seen.add(tid)
            uniq.append((tid, p))
    return uniq

async def main_async(args: argparse.Namespace):
    found = find_summary_files(args.scores_glob)
    if not found:
        print(f"[synth] No summary CSVs found with glob: {args.scores_glob}")
        return

    if args.shuffle:
        random.shuffle(found)
    if args.max_tasks and args.max_tasks > 0:
        found = found[:args.max_tasks]

    print(f"[synth] Found {len(found)} task summaries to process."); sys.stdout.flush()

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = ChatClient(session, timeout=args.timeout, retries=args.retries, backoff=args.backoff, live_prefix="[stream] ")
        sem_tasks = asyncio.Semaphore(max(1, args.task_concurrency))
        global_sem = asyncio.Semaphore(max(1, args.global_request_concurrency))
        print_lock = asyncio.Lock()

        async def _run_one(task_id: str, scores_path: Path):
            async with sem_tasks:
                try:
                    await synthesize_one(client, args, task_id, scores_path, global_sem, print_lock)
                except Exception as e:
                    async with print_lock:
                        print(f"[synth:{task_id}] ERROR {e}\n{traceback.format_exc()}"); sys.stdout.flush()

        await asyncio.gather(*[
            asyncio.create_task(_run_one(tid, path))
            for (tid, path) in found
        ])

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthesize Python programs + DSL ops from scored instruction sets (with telemetry).")
    # data roots
    p.add_argument("--root", type=Path, required=True, help="ARC data root (contains <split>/<task_id>.json)")
    p.add_argument("--split", type=str, choices=["training","evaluation"], default="training")
    p.add_argument("--candidates-root", type=Path, default=Path("/content/outputs"),
                   help="Root containing <split>/<task_id>.json candidates")
    # model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--guided-json", action="store_true")
    p.add_argument("--no-response-format", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=0.6)
    p.add_argument("--stream-model", action="store_true", help="Use OpenAI-style streaming (auto-disabled when --guided-json).")
    p.add_argument("--stream-live", action="store_true", help="When --stream-model, print buffered chunks at JSON boundaries.")
    # selection
    p.add_argument("--scores-glob", type=str, required=True,
                   help="Glob for candidates_summary.csv files (e.g. /content/debug_multi/**/candidates_summary.csv)")
    p.add_argument("--selection-policy", type=str, choices=["solved_only","fallback_highest"], default="fallback_highest",
                   help="Pick solved (100%) only, or fallback to highest accuracy if none solved.")
    p.add_argument("--max-instruction-sets", type=int, default=3,
                   help="How many top instruction sets to include in the prompt (K-best).")
    p.add_argument("--max-pairs-in-prompt", type=int, default=6)
    p.add_argument("--max-pairs-validate", type=int, default=8)
    p.add_argument("--max-tasks", type=int, default=0, help="Process at most N tasks (0=no limit)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle task order before limiting.")
    # concurrency
    p.add_argument("--task-concurrency", type=int, default=2)
    p.add_argument("--global-request-concurrency", type=int, default=8)
    # validation watchdog
    p.add_argument("--validation-timeout", type=int, default=20, help="Max seconds to validate code per task")
    # outputs
    p.add_argument("--out-dir", type=Path, required=True)
    # console prints
    p.add_argument("--print-code", action="store_true", help="Print synthesized Python program to stdout.")
    p.add_argument("--print-dsl", action="store_true", help="Print synthesized DSL operations to stdout.")
    p.add_argument("--print-prompt-preview", action="store_true", help="Print a preview of the prompt.")
    p.add_argument("--print-limit", type=int, default=2000, help="Max chars to print for long fields (0 = unlimited).")
    return p

def main():
    args = build_parser().parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()


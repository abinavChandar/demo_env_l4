
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARC end-to-end: select top instruction sets -> synthesize Python + DSL -> evaluate on training pairs.

Adds robust code sanitation:
  - Forbid imports in prompt; strip 'import ...' and 'from ... import ...' by default
  - Remove code fences and normalize newlines
  - Collapse/remove trailing backslash continuations; drop lone trailing '\'
  - Validate with compile() before exec(); print clear diagnostics and a sanitized preview
"""

from __future__ import annotations

import argparse
import aiohttp
import asyncio
import csv
import glob
import json
import os
import random
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Shared helpers ----------------

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
    out = []
    for p in pairs:
        x, y = p.get("input"), p.get("output")
        if not (is_valid_grid(x) and is_valid_grid(y)):
            raise ValueError(f"Invalid grids in {f}")
        out.append({"input": x, "output": y})
    return out

# ---------------- Scores / candidates ----------------

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
    stem = summary_path.stem  # "xxxxxx.candidates_summary"
    tid = stem.replace(".candidates_summary", "")
    parent = summary_path.parent.name
    if parent.endswith("_allc_out"):
        tid = parent[:-9] or tid
    return tid

def find_summary_files(scores_glob: str) -> List[Tuple[str, Path]]:
    raw: List[Tuple[str, Path]] = []
    for p in glob.glob(scores_glob, recursive=True):
        path = Path(p)
        if path.name.endswith("candidates_summary.csv"):
            tid = find_task_id_from_summary(path)
            raw.append((tid, path))
    # de-dup
    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for tid, p in raw:
        if tid not in seen:
            seen.add(tid)
            uniq.append((tid, p))
    return uniq

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
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(x.strip() for x in v if x.strip())
            if joined:
                return joined
    msgs = cand.get("messages")
    if isinstance(msgs, list) and msgs:
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            c = (isinstance(m, dict) and m.get("content"))
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
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    joined = "\n".join(x.strip() for x in v if x.strip())
                    if joined:
                        return joined
    return None

def pick_topk_with_tie_shuffle(scores: List[CandidateScore], k: int, rng: random.Random) -> List[int]:
    if not scores:
        return []
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

# ---------------- Prompt building ----------------

SYSTEM = (
    "You are ARC-Solver, an expert at Algorithmic Reasoning over Colored grids (ARC).\n"
    "Given several high-quality INSTRUCTION SETS (describing a single transformation rule),\n"
    "synthesize a deterministic Python module that defines def transform(grid): -> List[List[int]].\n"
    "Also propose a concise list of DSL primitives (name, signature, description) for this solution.\n"
    "Hard constraints for python_code:\n"
    "  - NO imports (do not use 'import' or 'from ... import').\n"
    "  - NO line continuation backslashes ('\\').\n"
    "  - Standard library only; deterministic; no I/O, prints, globals, or randomness.\n"
    "  - Grids are rectangular List[List[int]] with values 0..9; resize if rule demands it.\n"
    "  - Prefer small helpers (bbox, flood_fill, masks, draw_line, reflect/rotate, tile/upscale).\n"
    "  - Avoid overfitting; code must generalize to inputs consistent with the rule.\n"
)

def build_user_prompt(task_id: str,
                      instruction_texts: List[str],
                      pairs: Optional[List[Dict[str, Any]]] = None,
                      include_pairs: bool = False,
                      max_pairs: int = 0) -> str:
    lines = [f"TASK {task_id}", ""]
    if include_pairs and pairs:
        subset = pairs if max_pairs <= 0 else pairs[:max_pairs]
        lines.append("TRAINING EXAMPLES (INPUT -> OUTPUT) as JSON arrays:")
        for i, p in enumerate(subset):
            x = json.dumps(p["input"], separators=(",", ":"))
            y = json.dumps(p["output"], separators=(",", ":"))
            lines.append(f"- Pair {i}: INPUT={x}  OUTPUT={y}")
        lines.append("")

    lines.append("CANDIDATE INSTRUCTION SETS (top 3; ties randomly broken):")
    for i, instr in enumerate(instruction_texts, 1):
        lines.append(f"\n--- Candidate #{i} ---\n{instr.strip()}")

    lines.append("""
REQUIRED OUTPUT FORMAT (STRICT JSON, no code fences):
{
  "python_code": "<full module text that defines def transform(grid): ...>",
  "dsl_operations": [
    {"name": "op_name", "signature": "op(arg1: T, arg2: U) -> V", "description": "what it does"},
    ...
  ],
  "rationale": "<optional short reasoning>"
}

Notes:
- Do NOT include 'import' or 'from ... import' statements. Do NOT use '\\' line continuations.
- transform(grid) must accept/return List[List[int]]. Validate shapes and values (0..9) as needed.
- Prefer clear, composable helpers over monoliths.
""")
    return "\n".join(lines)

# ---------------- JSON salvage ----------------

def salvage_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    # strip fenced blocks
    if s.startswith("```"):
        fence_end = s.rfind("```")
        if fence_end > 3:
            s = s[3:fence_end].strip()
            s = s.split("\n", 1)[-1].strip() if "\n" in s else s
    # bracket slice
    try:
        i = s.index("{")
        j = s.rfind("}")
        if j <= i:
            return None
        snippet = s[i:j+1]
        return json.loads(snippet)
    except Exception:
        return None

# ---------------- Model client ----------------

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
                                "description": {"type": "string"}
                            },
                            "required": ["name", "signature", "description"],
                            "additionalProperties": False
                        },
                        "minItems": 1
                    },
                    "rationale": {"type": "string"}
                },
                "required": ["python_code", "dsl_operations"],
                "additionalProperties": False
            },
            "strict": True
        }

    async def generate(self, *, model: str, system: str, user: str, max_tokens: int,
                       use_guided_json: bool, use_response_format: bool) -> GenResult:
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

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                async with self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout) as resp:
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
            except Exception as e:
                last_err = f"request error: {e}"
            if attempt < self.retries:
                await asyncio.sleep(self.backoff * (attempt + 1))
        return GenResult(text=None, error=last_err)

# ---------------- Code sanitation ----------------

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        end = s.rfind("```")
        if end > 3:
            s = s[3:end].strip()
            # drop optional language tag line
            if "\n" in s:
                first, rest = s.split("\n", 1)
                # if first looks like a language tag
                if re.fullmatch(r"[A-Za-z0-9_+\-\. ]{1,40}", first.strip()):
                    s = rest
                else:
                    s = first + "\n" + rest
    return s

def sanitize_python_code(code: str, *, allow_imports: bool = False) -> Tuple[str, List[str]]:
    """
    Returns (sanitized_code, notes)
    - removes fences, normalizes \r\n
    - strips imports unless allow_imports
    - collapses/removes trailing backslashes
    - ensures final newline
    """
    notes: List[str] = []
    if not code:
        return code, ["empty python_code"]

    code = strip_code_fences(code).replace("\r\n", "\n").replace("\r", "\n")

    new_lines: List[str] = []
    lines = code.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        # strip imports if not allowed
        if not allow_imports and re.match(r"^\s*(from\s+\w|import\s+)", line):
            notes.append(f"stripped import: {line.strip()}")
            i += 1
            continue

        # handle trailing backslash continuation
        if line.rstrip().endswith("\\"):
            # join with subsequent non-empty lines until no trailing '\'
            buf = line.rstrip().rstrip("\\").rstrip()
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                # skip blank lines between continuation
                j += 1
            while j < len(lines) and lines[j].rstrip().endswith("\\"):
                buf += " " + lines[j].rstrip().rstrip("\\").strip()
                j += 1
            if j < len(lines):
                buf += " " + lines[j].strip()
                new_lines.append(buf)
                notes.append("collapsed line-continuation backslashes")
                i = j + 1
                continue
            else:
                # dangling backslash at EOF: drop it
                new_lines.append(buf)
                notes.append("dropped dangling trailing backslash at EOF")
                i = j
                continue

        new_lines.append(line)
        i += 1

    sanitized = "\n".join(new_lines)
    if not sanitized.endswith("\n"):
        sanitized += "\n"

    return sanitized, notes

# ---------------- Build & eval per task ----------------

def build_transform_callable(python_code: str):
    """Exec synthesized module text and return transform()."""
    module = types.ModuleType("synth_module")
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

# ---------------- Task pipeline ----------------

async def synth_and_eval_task(task_id: str,
                              summary_path: Path,
                              args: argparse.Namespace,
                              client: ChatClient,
                              rng: random.Random,
                              per_pair_writer: Optional[csv.writer],
                              summary_writer: Optional[csv.writer]) -> None:
    print(f"\n=== Task {task_id} ===")
    print(f"summary: {summary_path}")

    scores = parse_summary_csv(summary_path)
    if not scores:
        print("(no rows in summary)")
        return

    chosen = pick_topk_with_tie_shuffle(scores, k=3, rng=rng)
    leaderboard = ", ".join(f"{s.candidate_index}:{s.mean_cell_accuracy:.2f}%" for s in sorted(scores, key=lambda s: s.mean_cell_accuracy, reverse=True)[:6])
    print(f"leaderboard (top): {leaderboard}")
    print(f"selected indices: {chosen}")

    # load candidates/instruction texts
    try:
        cands = read_candidates(Path(args.candidates_root), args.split, task_id)
    except Exception as e:
        print(f"ERROR reading candidates for {task_id}: {e}")
        return

    instruction_texts: List[str] = []
    for ci in chosen:
        if 0 <= ci < len(cands):
            txt = extract_instruction_text(cands[ci])
            if txt:
                instruction_texts.append(txt)
    if not instruction_texts:
        print("(no instruction texts found for selected indices)")
        return

    # optionally include training pairs in prompt
    pairs: Optional[List[Dict[str, Any]]] = None
    if args.include_pairs_in_prompt:
        try:
            pairs = load_arc_pairs(Path(args.root), args.split, task_id)
        except Exception as e:
            print(f"ERROR reading training pairs for {task_id}: {e}")
            return

    # BUILD PROMPT
    prompt = build_user_prompt(
        task_id,
        instruction_texts,
        pairs=pairs,
        include_pairs=bool(args.include_pairs_in_prompt),
        max_pairs=args.max_pairs_in_prompt,
    )

    print("\n--- FULL PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    # MODEL CALL
    t0 = time.time()
    print(f"[gen:{task_id}] ▶ model call (max_tokens={args.max_tokens})"); sys.stdout.flush()
    res = await client.generate(
        model=args.model,
        system=SYSTEM,
        user=prompt,
        max_tokens=args.max_tokens,
        use_guided_json=bool(args.guided_json),
        use_response_format=not args.no_response_format,
    )
    dt = time.time() - t0
    print(f"[gen:{task_id}] ◀ model done  wall={dt:.2f}s  chars={len((res.text or ''))}  err={res.error is not None}")
    raw = (res.text or "").strip()

    # PARSE JSON (with salvage)
    parsed: Optional[Dict[str, Any]] = None
    parse_err = None
    try:
        parsed = json.loads(raw)
    except Exception as e:
        parsed = salvage_json_object(raw)
        if parsed is None:
            parse_err = f"json parse error (and salvage failed): {e}"

    if parsed is None:
        print(f"[model] note: {parse_err}")
        print("[model] RAW (first 2000 chars):")
        print(raw[:2000])
        return

    dsl = parsed.get("dsl_operations")
    code_raw = parsed.get("python_code") or ""
    print("[model] DSL OPERATIONS:")
    try:
        print(json.dumps(dsl, ensure_ascii=False, indent=2))
    except Exception:
        print(dsl)
    print("\n[model] PYTHON PROGRAM (raw):")
    print(code_raw or "<missing python_code>")
    print()

    # SAVE raw synth JSON (optional)
    synth_fp = None
    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        synth_fp = out_dir / f"{task_id}.synth.json"
        with synth_fp.open("w", encoding="utf-8") as fh:
            json.dump(parsed, fh, ensure_ascii=False, indent=2)
        print(f"[save] wrote {synth_fp}")

    # SANITIZE CODE
    code_sanitized, notes = sanitize_python_code(code_raw, allow_imports=args.allow_imports)
    if notes:
        print("[sanitize] notes:")
        for n in notes:
            print(f"  - {n}")

    # quick diff-ish preview when changed
    if code_sanitized != code_raw:
        print("\n[sanitize] sanitized code preview (first 40 lines):")
        for i, line in enumerate(code_sanitized.splitlines()[:40], 1):
            print(f"{i:02d}: {line}")
        print()

    # ensure it defines transform
    if "def transform" not in code_sanitized:
        print(f"[eval:{task_id}] build error: synthesized code lacks def transform(grid)")
        if summary_writer:
            summary_writer.writerow([task_id, "0.0000", "missing_transform"])
        return

    # compile before exec
    try:
        compile(code_sanitized, "<synth>", "exec")
    except SyntaxError as e:
        print(f"[eval:{task_id}] compile error: {e!r}")
        # last-ditch: if error looks like EOF, append final newline
        if "EOF" in str(e):
            code_sanitized += "\n"
            try:
                compile(code_sanitized, "<synth>", "exec")
            except SyntaxError as e2:
                print(f"[eval:{task_id}] compile error after newline: {e2!r}")
                if summary_writer:
                    summary_writer.writerow([task_id, "0.0000", "compile_error"])
                return
        else:
            if summary_writer:
                summary_writer.writerow([task_id, "0.0000", "compile_error"])
            return

    # EVALUATE on training pairs
    try:
        all_pairs = pairs if pairs is not None else load_arc_pairs(Path(args.root), args.split, task_id)
    except Exception as e:
        print(f"ERROR reading training pairs for eval {task_id}: {e}")
        return

    try:
        transform = build_transform_callable(code_sanitized)
    except Exception as e:
        print(f"[eval:{task_id}] exec/build error: {e!r}")
        if summary_writer:
            summary_writer.writerow([task_id, "0.0000", "exec_error"])
        return

    total = 0.0
    for i, p in enumerate(all_pairs):
        x, y = p["input"], p["output"]
        note = "ok"
        pred = None
        try:
            pred = transform([row[:] for row in x])
            if not is_valid_grid(pred):
                note = "invalid grid"
                pred = None
        except Exception as e:
            note = f"runtime error: {e!r}"
            pred = None
        acc = cell_accuracy(pred, y)
        total += acc
        print(f"[Pair {i}] ACC={acc:.2f}%  note={note}")
        if args.print_grids:
            def g2s(g): return "\n".join(" ".join(str(v) for v in row) for row in g)
            print("INPUT:");  print(g2s(x))
            print("TARGET:"); print(g2s(y))
            print("PRED  :"); print("<None>" if pred is None else g2s(pred))
            print("-" * 60)
        if per_pair_writer:
            per_pair_writer.writerow([
                task_id, i, f"{acc:.4f}", note,
                json.dumps(x, separators=(",", ":")),
                json.dumps(y, separators=(",", ":")),
                json.dumps(pred, separators=(",", ":")) if pred is not None else "",
            ])

    mean_acc = total / len(all_pairs) if all_pairs else 0.0
    print(f"[Task {task_id}] MEAN_CELL_ACC = {mean_acc:.2f}%")
    if summary_writer:
        summary_writer.writerow([task_id, f"{mean_acc:.4f}", "ok"])

# ---------------- Orchestration ----------------

async def main_async(args: argparse.Namespace):
    rng = random.Random(args.seed)

    # discover tasks
    found = find_summary_files(args.scores_glob)
    if args.shuffle:
        rng.shuffle(found)
    if args.max_tasks and args.max_tasks > 0:
        found = found[:args.max_tasks]
    if not found:
        print(f"[run] No summary CSVs found with glob: {args.scores_glob}")
        return
    print(f"[run] Found {len(found)} tasks.")

    # optional CSV writers
    per_pair_writer = None
    summary_writer = None
    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        per_pair_fp = out_dir / "per_pair.csv"
        summary_fp = out_dir / "summary.csv"
        per_pair_writer = csv.writer(per_pair_fp.open("w", newline="", encoding="utf-8"))
        summary_writer = csv.writer(summary_fp.open("w", newline="", encoding="utf-8"))
        per_pair_writer.writerow(["task_id","pair_index","cell_accuracy","note","input","target","pred"])
        summary_writer.writerow(["task_id","mean_cell_accuracy","status"])

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = ChatClient(session, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
        # sequential per task (safer for long prompts). Parallelize if you like.
        for tid, spath in found:
            await synth_and_eval_task(
                tid, spath, args, client, rng,
                per_pair_writer=per_pair_writer,
                summary_writer=summary_writer
            )

    print("\n[run] Done.")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ARC: synthesize Python+DSL from instruction sets and evaluate on training pairs (with sanitization).")
    # discovery / selection
    p.add_argument("--scores-glob", type=str, required=True,
                   help='Glob for candidates_summary.csv (e.g. "/content/debug_multi/**/*candidates_summary.csv")')
    p.add_argument("--candidates-root", type=Path, default=Path("/content/outputs"),
                   help="Root containing <split>/<task_id>.json (candidates)")
    p.add_argument("--root", type=Path, required=True, help="ARC data root (contains <split>/<task_id>.json)")
    p.add_argument("--split", type=str, choices=["training","evaluation"], default="training")
    p.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaking among equal scores")
    p.add_argument("--max-tasks", type=int, default=0, help="Limit number of tasks (0 = all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle task order before limiting")
    # prompt content
    p.add_argument("--include-pairs-in-prompt", action="store_true", help="If set, include training pairs in the prompt")
    p.add_argument("--max-pairs-in-prompt", type=int, default=0, help="0 = include all training pairs (when included)")
    # model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--guided-json", action="store_true")
    p.add_argument("--no-response-format", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=0.6)
    # output & printing
    p.add_argument("--out-dir", type=Path, default=None, help="Directory to save synth JSON + eval CSVs")
    p.add_argument("--print-grids", action="store_true", help="Print grids for each evaluated pair")
    # sanitation
    p.add_argument("--allow-imports", action="store_true", help="Do NOT strip imports from python_code")
    return p

def main():
    args = build_parser().parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()


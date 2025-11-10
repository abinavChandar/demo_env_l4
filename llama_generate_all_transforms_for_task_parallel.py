#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw-Python ARC transform generator (robust, self-healing).

Key features:
- Accepts NL instruction candidates JSON (and optional ARC IO pairs for reference).
- Prompts for code-only; enforces exact header/signature.
- Auto-coerces model output into a valid file if header/signature missing.
- Allows only a small dsl_* op subset; static checks + one auto-repair pass.
- Endpoint fallback: tries /v1/chat/completions, then /v1/completions.
- Auto-resolves model id from /v1/models (or /models) when possible.
- Parallel generation via ThreadPoolExecutor.

Expected input JSON shapes:
  { "candidates":[{"instructions":"..."}, ...], "train":[{"input":...,"output":...}, ...] }
  [{"instructions":"..."}, ...]
  { "instructions":"..." }
"""

import argparse, os, sys, json, re, ast, traceback, concurrent.futures as cf
from typing import Any, Dict, List, Optional

# ============================ Constants & Config ============================

HEADER = (
    "from typing import List\n"
    "import dsl_primitives as dsl\n\n"
)
SIG_LINE = "def transform(grid: List[List[int]]) -> List[List[int]]:"

HEADER_SNIPPET = (
    HEADER +
    SIG_LINE + "\n"
    "    # write solution below\n"
    "    G = dsl.dsl_clone(grid)\n"
    "    return G\n"
)

DEFAULT_ALLOWED_OPS = [
    # utils / shape
    "dsl_shape", "dsl_clone", "dsl_clamp", "dsl_zeros_like", "dsl_full", "dsl_in_bounds",
    # accessors
    "dsl_get_cell", "dsl_set_cell", "dsl_paint_cell",
    # colors
    "dsl_replace_color", "dsl_remap_colors",
    # rows/cols
    "dsl_paint_row", "dsl_paint_col", "dsl_copy_row", "dsl_copy_col",
    # geometry
    "dsl_flip_h", "dsl_flip_v", "dsl_transpose", "dsl_rot90",
    # regions
    "dsl_fill_rect", "dsl_paste", "dsl_paste_masked", "dsl_crop",
    # masks
    "dsl_mask_eq", "dsl_apply_mask_color", "dsl_dilate", "dsl_erode",
    # connected components
    "dsl_neighbors4", "dsl_flood_fill", "dsl_component_mask", "dsl_bbox_of_mask",
    # composition
    "dsl_copy_cell_from", "dsl_write_component",
]

STOP_SEQUENCES = ["```", "</code>", "###", "```python"]

FORBIDDEN_PATTERNS = [
    r"^\s*import\s+(?!dsl_primitives\b)",  # no extra imports (only allowed header)
    r"\bopen\s*\(",
    r"\bprint\s*\(",
    r"\bos\.|subprocess|socket|sys\.stdin|sys\.stdout|random\b|torch\b|numpy\b",
]

SIGNATURE_RE = re.compile(
    r"from\s+typing\s+import\s+List\s*[\r\n]+"
    r"import\s+dsl_primitives\s+as\s+dsl\s*[\r\n]+"
    r"def\s+transform\s*\(\s*grid\s*:\s*List\s*\[\s*List\s*\[\s*int\s*\]\s*\]\s*\)"
    r"\s*->\s*List\s*\[\s*List\s*\[\s*int\s*\]\s*\]\s*:",
    re.IGNORECASE | re.MULTILINE
)

# ============================ Extractors ============================

def extract_nl_candidates(obj: Any) -> List[str]:
    """NL candidates:
       - {"candidates":[{"instructions":"..."} , ...]}
       - [{"instructions":"..."}, ...]
       - {"instructions":"..."}  (single)
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        if "candidates" in obj and isinstance(obj["candidates"], list):
            return [x.get("instructions", "") for x in obj["candidates"] if isinstance(x, dict)]
        if isinstance(obj.get("instructions"), str):
            return [obj["instructions"]]
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict) and isinstance(x.get("instructions"), str):
                out.append(x["instructions"])
        return out
    return []

def extract_io_pairs(obj: Any) -> List[Dict[str, Any]]:
    """ARC IO pairs for reference (optional):
       - {"train":[{"input":..,"output":..}, ...]}
       - {"io_pairs":[...]} or {"pairs":[...]} or {"examples":[...]}
       - direct list of {"input","output"}
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        for key in ("io_pairs", "pairs", "examples"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if isinstance(obj.get("train"), list):
            res = []
            for ex in obj["train"]:
                if isinstance(ex, dict) and "input" in ex and "output" in ex:
                    res.append({"input": ex["input"], "output": ex["output"]})
            return res
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and \
       ("input" in obj[0] and "output" in obj[0]):
        return obj
    return []

# ============================ Prompt Builders ============================

def build_system_prompt() -> str:
    return (
        "You write Python for ARC grid transforms using a tiny DSL already imported as `dsl`.\n"
        "Output ONLY one Python file with EXACTLY this header and a single function.\n"
        "No prose, no markdown, no extra text — code only.\n"
    )

def build_user_prompt(nl_text: str,
                      io_pairs: List[Dict[str, Any]],
                      allowed_ops: List[str]) -> str:
    parts = []
    parts.append("Output ONLY the code between <BEGIN_FILE> and <END_FILE>. No prose.")
    parts.append("Your file MUST start exactly with this header+signature:")
    parts.append("<HEADER>\n" + HEADER + SIG_LINE + "\n    # write solution below\n</HEADER>")
    parts.append("Allowed ops subset (use only these): " + ", ".join(allowed_ops))
    parts.append("Task instructions:\n" + nl_text.strip())
    if io_pairs:
        parts.append("# reference_examples = " + json.dumps(io_pairs[:2], ensure_ascii=False))
    # Provide a scaffold they can overwrite:
    parts.append("<BEGIN_FILE>\n" + HEADER_SNIPPET + "\n<END_FILE>")
    return "\n\n".join(parts)

def build_repair_prompt(code: str, error: str) -> str:
    return (
        "Your previous code failed to compile or run. Fix it WITHOUT changing the header imports or the function name.\n"
        "Keep ONLY the single function in the file. Output ONLY the corrected code (no prose).\n\n"
        "<BEGIN_PREVIOUS_CODE>\n" + code + "\n<END_PREVIOUS_CODE>\n\n"
        "Error (first lines):\n" + error.strip() + "\n\n"
        "Re-output the FULL corrected file, code ONLY, starting with the required header."
    )

# ============================ Coercion & Checks ============================

def extract_code(text: str) -> str:
    """Strip markdown fences if present; otherwise return raw text."""
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # If <BEGIN_FILE> guard is present, use it
    g = re.search(r"<BEGIN_FILE>\s*([\s\S]*?)\s*<END_FILE>", text, re.IGNORECASE)
    if g:
        return g.group(1).strip()
    return text.strip()

def _extract_body_or_return(code: str) -> str:
    """Try to pull the body of transform() from arbitrary text/code."""
    # 1) fenced code
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", code, re.IGNORECASE)
    if m:
        code = m.group(1)
    # 2) find 'def transform(...):' and capture its block
    m = re.search(r"def\s+transform\s*\([^)]*\)\s*:\s*([\s\S]*)", code)
    if not m:
        return ""
    body = m.group(1)
    lines = body.splitlines()
    if not lines or all(not ln.strip() for ln in lines):
        return "    pass\n"
    # If first content line is not indented, indent best-effort
    if lines and lines[0] and not lines[0].startswith((" ", "\t")):
        lines = ["    " + ln for ln in lines]
    return "\n".join(lines).rstrip() + ("\n" if not body.endswith("\n") else "")

def coerce_to_valid_file(text: str) -> str:
    """Guarantee header+signature; wrap extracted body or a safe stub."""
    code = extract_code(text)
    if "from typing import List" in code and "import dsl_primitives as dsl" in code and "def transform(" in code:
        return code.strip()
    body = _extract_body_or_return(code)
    if not body:
        body = "    G = dsl.dsl_clone(grid)\n    return G\n"
    if "return " not in body:
        body = "    G = dsl.dsl_clone(grid)\n" + body + "\n    return G\n"
    return f"{HEADER}{SIG_LINE}\n{body}"

def static_checks(code: str, allowed_ops: List[str]) -> Optional[str]:
    """Return None if ok, else error string describing the violation."""
    if not SIGNATURE_RE.search(code):
        return "Missing or incorrect header/signature."
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return f"Forbidden pattern matched: {pat}"
    calls = re.findall(r"\bdsl\.(\w+)\s*\(", code)
    for c in calls:
        if not c.startswith("dsl_"):
            return f"Non-DSL call detected: {c}"
        if c not in allowed_ops:
            return f"Call to disallowed op: dsl.{c}"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    return None

def runtime_smoke(code_path: str) -> Optional[str]:
    """Import the produced module and run a minimal smoke test."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("transform_module", code_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "transform"):
            return "No transform() defined after import."
        out = mod.transform([[0]])
        if not isinstance(out, list) or (out and not isinstance(out[0], list)):
            return "transform did not return List[List[int]]"
        return None
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"Runtime import/run error: {tb}"

# ============================ HTTP & Model Resolve ============================

def list_models(api_base: str, timeout=(5.0, 10.0)) -> list:
    import requests
    urls = [f"{api_base.rstrip('/')}/v1/models", f"{api_base.rstrip('/')}/models"]
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                js = r.json()
                if isinstance(js, dict) and "data" in js:
                    return [m.get("id") for m in js["data"] if isinstance(m, dict)]
                if isinstance(js, list):
                    return [m.get("id", m) if isinstance(m, dict) else m for m in js]
        except Exception:
            pass
    return []

def resolve_model_id(requested: str, available: list) -> str:
    if not available or requested in available:
        return requested
    import re
    def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
    nreq = norm(requested)
    for a in available:
        if norm(a) == nreq:
            return a
    for a in available:
        if nreq in norm(a) or norm(a) in nreq:
            return a
    return available[0]

def chat_completion(api_base: str,
                    model: str,
                    system_prompt: str,
                    user_prompt: str,
                    temperature: float,
                    max_tokens: int,
                    connect_timeout: float,
                    read_timeout: float) -> str:
    import requests
    def _post_json(url, body):
        return requests.post(
            url, json=body,
            headers={"Content-Type": "application/json"},
            timeout=(connect_timeout, read_timeout),
        )
    # Try Chat Completions
    chat_url = f"{api_base.rstrip('/')}/v1/chat/completions"
    chat_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": STOP_SEQUENCES,
    }
    r = _post_json(chat_url, chat_body)
    if r.status_code in (404, 405):
        # Fallback: Text Completions
        comp_url = f"{api_base.rstrip('/')}/v1/completions"
        prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"
        comp_body = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": STOP_SEQUENCES,
        }
        r2 = _post_json(comp_url, comp_body)
        try:
            r2.raise_for_status()
        except Exception:
            raise RuntimeError(
                f"Both endpoints failed. chat/completions={r.status_code} {r.text[:300]!r}; "
                f"completions={r2.status_code} {r2.text[:300]!r}"
            )
        payload = r2.json()
        return payload["choices"][0].get("text", "")
    r.raise_for_status()
    payload = r.json()
    return payload["choices"][0]["message"]["content"]

# ============================ Worker ============================

def _write(path: str, text: str) -> Optional[str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def process_candidate(idx: int,
                      args,
                      nl_text: str,
                      io_pairs_for_prompt: List[Dict[str, Any]],
                      allowed_ops: List[str],
                      out_dir: str) -> Dict[str, Any]:
    py_path = os.path.join(out_dir, f"{args.task_id}.cand_{idx}.transform.nl.py")
    log_path = os.path.join(out_dir, f"{args.task_id}.cand_{idx}.log.json")
    result = {"candidate": idx, "status": "unknown", "py_path": py_path}

    try:
        sys_prompt = build_system_prompt()
        usr_prompt = build_user_prompt(nl_text, io_pairs_for_prompt, allowed_ops)

        # 1) Generate code
        raw = chat_completion(
            api_base=args.api_base,
            model=args.model,
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            connect_timeout=args.http_connect_timeout,
            read_timeout=args.http_read_timeout,
        )
        code = coerce_to_valid_file(raw)

        # 2) Pre-flight checks
        err = static_checks(code, allowed_ops)
        if err:
            # 3) Auto-repair once
            repair_prompt = build_repair_prompt(code, err)
            raw2 = chat_completion(
                api_base=args.api_base,
                model=args.model,
                system_prompt=sys_prompt,
                user_prompt=repair_prompt,
                temperature=min(args.temperature + 0.1, 0.7),
                max_tokens=args.max_tokens,
                connect_timeout=args.http_connect_timeout,
                read_timeout=args.http_read_timeout,
            )
            code2 = coerce_to_valid_file(raw2)
            err2 = static_checks(code2, allowed_ops)
            if err2:
                result["status"] = "invalid_code"
                result["error"] = f"Pre-flight failed after repair: {err2}"
                _write(py_path, code2)
                _write(log_path, json.dumps(result, indent=2))
                return result
            code = code2

        # 4) Write & smoke test
        _write(py_path, code)
        rterr = runtime_smoke(py_path)
        if rterr:
            # one more chance: repair with runtime error
            repair_prompt = build_repair_prompt(code, rterr)
            raw3 = chat_completion(
                api_base=args.api_base,
                model=args.model,
                system_prompt=sys_prompt,
                user_prompt=repair_prompt,
                temperature=min(args.temperature + 0.1, 0.7),
                max_tokens=args.max_tokens,
                connect_timeout=args.http_connect_timeout,
                read_timeout=args.http_read_timeout,
            )
            code3 = coerce_to_valid_file(raw3)
            _write(py_path, code3)
            err3 = static_checks(code3, allowed_ops)
            if err3:
                result["status"] = "runtime_error"
                result["error"] = f"Runtime failed after repair: {err3}"
                _write(log_path, json.dumps(result, indent=2))
                return result
            rterr2 = runtime_smoke(py_path)
            if rterr2:
                result["status"] = "runtime_error"
                result["error"] = f"Runtime failed after repair: {rterr2}"
                _write(log_path, json.dumps(result, indent=2))
                return result

        result["status"] = "ok"
        _write(log_path, json.dumps(result, indent=2))
        return result

    except Exception as e:
        result["status"] = "api_or_unexpected_error"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _write(log_path, json.dumps(result, indent=2))
        return result

# ============================ Main ============================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-root", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--instr-json", required=True, help="Path to NL candidates JSON (and/or ARC IO pairs)")
    p.add_argument("--api-base", required=True)
    p.add_argument("--model", required=True)

    # Concurrency & decoding
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--max-tokens", type=int, default=400)

    # HTTP timeouts
    p.add_argument("--http-connect-timeout", type=float, default=10.0)
    p.add_argument("--http-read-timeout", type=float, default=600.0)

    # Allowlist override (comma-separated)
    p.add_argument("--allowed-ops", default=",".join(DEFAULT_ALLOWED_OPS))

    # Candidate cap (optional)
    p.add_argument("--max-candidates", type=int, default=None)

    args = p.parse_args()

    # Load input JSON
    with open(args.instr_json, "r", encoding="utf-8") as f:
        blob = json.load(f)

    nl_list = extract_nl_candidates(blob)
    io_pairs = extract_io_pairs(blob)[:2]  # keep prompt short

    if not nl_list:
        print("[FATAL] No NL instruction candidates found. Expected {'candidates':[{'instructions': ...}]} or a list of such dicts.", file=sys.stderr)
        sys.exit(2)
    if args.max_candidates is not None:
        nl_list = nl_list[: args.max_candidates]

    # Resolve model id if the server has a slightly different name
    avail = list_models(args.api_base)
    if avail:
        resolved = resolve_model_id(args.model, avail)
        if resolved != args.model:
            print(f"[INFO] Model '{args.model}' not found; using '{resolved}' from server list: {avail}")
            args.model = resolved
    else:
        print("[WARN] Could not list models from server; proceeding with requested id:", args.model)

    allowed_ops = [x.strip() for x in args.allowed_ops.split(",") if x.strip()]
    out_dir = os.path.join(args.scores_root, f"{args.task_id}_allc_out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Task {args.task_id}: {len(nl_list)} candidate(s); workers={args.workers}")
    print(f"[INFO] Using {len(io_pairs)} IO pair(s) for reference in prompt.")
    print(f"[INFO] Allowed ops: {', '.join(allowed_ops)}")

    # Run in parallel
    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(process_candidate, i, args, nl_list[i], io_pairs, allowed_ops, out_dir)
            for i in range(len(nl_list))
        ]
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            print("=" * 75)
            print(f"[RESULT] Task {args.task_id} | Candidate {r.get('candidate')} | status={r.get('status')}")
            if r.get("status") == "ok":
                print(f"Python: {r['py_path']}")
            else:
                print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
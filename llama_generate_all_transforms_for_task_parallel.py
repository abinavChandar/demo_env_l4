
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, ast, glob, json, os, re, sys, time, uuid, random, threading, inspect, importlib
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ------------------------ logging & telemetry ------------------------

def log(level: str, msg: str):
    print(f"[{level}] {msg}", flush=True)

class Telemetry:
    def __init__(self, scores_root: str, emit_events: bool):
        self.start_ts = time.time()
        self.run_id = f"nlgen_all_par_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.emit_events = emit_events
        self.root_dir = os.path.join(scores_root, "telemetry")
        os.makedirs(self.root_dir, exist_ok=True)
        self.summary_path = os.path.join(self.root_dir, f"{self.run_id}.summary.json")
        self.events_path  = os.path.join(self.root_dir, f"{self.run_id}.events.jsonl")
        self.events_lock = threading.Lock()
        if self.emit_events:
            open(self.events_path, "w").close()
        self.rollup = {
            "run_id": self.run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.start_ts)),
            "dur_seconds": None,
            "scores_root": scores_root,
            "task_id": None,
            "candidates_total": 0,
            "candidates_ok": 0,
            "candidates_err": 0,
            "api_base": None,
            "model": None,
            "temperature": None,
            "max_tokens": None,
            "workers": None,
            "notes": [],
        }

    def event(self, kind: str, **payload):
        if not self.emit_events: return
        rec = {"ts": time.time(), "run_id": self.run_id, "kind": kind, **payload}
        with self.events_lock:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def end(self):
        self.rollup["dur_seconds"] = round(time.time() - self.start_ts, 6)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.rollup, f, indent=2)

# ------------------------ constants ------------------------

INSTR_COLS_JSON = ["instructions","instruction","instructions_text","text","plan","nl"]
CAND_ID_COLS_JSON = ["candidate_index","candidate_idx","cand_idx","cand_index","index","idx","id","candidate_id","cand_id"]

SAFE_BUILTIN_CALLS: Set[str] = {
    "range","len","enumerate","list","tuple","int","max","min","sum","any","all","zip"
}

# ------------------------ helpers: parsing candidates ------------------------

def split_lines(s: str) -> List[str]:
    parts = re.split(r"[\n;]+", s)
    out: List[str] = []
    for p in parts:
        p = re.sub(r"^\s*[\-\*\d\.\)\(]+\s*", "", p.strip())
        if p: out.append(p)
    return out

def normalize_instructions(val: Any) -> List[str]:
    if val is None: return []
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            if x is None: continue
            s = str(x).strip()
            if not s: continue
            out.extend(split_lines(s))
        return out
    s = str(val)
    if "[" in s and "]" in s:
        try:
            arr = json.loads(s)
            if isinstance(arr, list): return normalize_instructions(arr)
        except Exception:
            pass
    return split_lines(s)

def enumerate_candidates(instr_json_path: str) -> List[Tuple[str,int,List[str]]]:
    if not os.path.exists(instr_json_path):
        raise FileNotFoundError(f"Instructions JSON not found: {instr_json_path}")
    with open(instr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Tuple[str,int,List[str]]] = []

    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        for i, cand in enumerate(data["candidates"]):
            if not isinstance(cand, dict): continue
            cid = None
            for k in CAND_ID_COLS_JSON:
                if k in cand and cand[k] is not None:
                    cid = str(cand[k]).strip(); break
            instr_raw = None
            for k in INSTR_COLS_JSON:
                if k in cand and cand[k] is not None:
                    instr_raw = cand[k]; break
            instr = normalize_instructions(instr_raw)
            if instr: out.append((cid if cid is not None else str(i), i, instr))
        return out

    if isinstance(data, list):
        for i, cand in enumerate(data):
            if isinstance(cand, dict):
                cid = None
                for k in CAND_ID_COLS_JSON:
                    if k in cand and cand[k] is not None:
                        cid = str(cand[k]).strip(); break
                instr_raw = None
                for k in INSTR_COLS_JSON:
                    if k in cand and cand[k] is not None:
                        instr_raw = cand[k]; break
                instr = normalize_instructions(instr_raw)
                if instr: out.append((cid if cid is not None else str(i), i, instr))
            elif isinstance(cand, str):
                out.append((str(i), i, normalize_instructions(cand)))
        return out

    if isinstance(data, dict):
        single = None
        for k in INSTR_COLS_JSON:
            if k in data and data[k] is not None:
                single = normalize_instructions(data[k]); break
        if single: out.append(("0", 0, single))
        if isinstance(data.get("by_id"), dict):
            for cid, obj in data["by_id"].items():
                if not isinstance(obj, dict): continue
                instr_raw = None
                for k in INSTR_COLS_JSON:
                    if k in obj and obj[k] is not None:
                        instr_raw = obj[k]; break
                instr = normalize_instructions(instr_raw)
                if instr: out.append((str(cid), len(out), instr))
        if isinstance(data.get("by_index"), list):
            for i, obj in enumerate(data["by_index"]):
                if not isinstance(obj, dict): continue
                instr_raw = None
                for k in INSTR_COLS_JSON:
                    if k in obj and obj[k] is not None:
                        instr_raw = obj[k]; break
                instr = normalize_instructions(instr_raw)
                if instr: out.append((str(i), i, instr))
        seen=set(); dedup=[]
        for cid,i,instr in out:
            if cid in seen: continue
            seen.add(cid); dedup.append((cid,i,instr))
        return dedup

    return out

# ------------------------ ARC loading ------------------------

def load_arc_task(arc_root: str, task_id: str) -> dict:
    p = os.path.join(arc_root, f"{task_id}.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    hits = glob.glob(os.path.join(arc_root, "**", f"{task_id}.json"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"ARC file not found for {task_id} under {arc_root}")
    with open(sorted(hits)[0], "r", encoding="utf-8") as f:
        return json.load(f)

def pick_train_shots(task_obj: dict, shots: int, seed: Optional[int]) -> List[dict]:
    train = task_obj.get("train", []) or []
    if seed is not None: random.seed(seed)
    if shots <= 0 or shots >= len(train): return train
    return random.sample(train, shots)

# ------------------------ DSL discovery & prompts ------------------------

def discover_dsl_ops(dsl_module: str) -> List[str]:
    try:
        mod = importlib.import_module(dsl_module)
        ops = []
        for name, fn in mod.__dict__.items():
            if callable(fn) and name.startswith("dsl_"):
                try:
                    sig = str(inspect.signature(fn))
                except Exception:
                    sig = "(...)"
                ops.append(f"{name}{sig}")
        ops.sort()
        return ops
    except Exception as e:
        log("WARN", f"Could not import DSL module '{dsl_module}': {e}")
        return []

def build_system_prompt(dsl_module: str, dsl_ops: List[str]) -> str:
    op_list = "\n".join(f"- {s}" for s in dsl_ops) if dsl_ops else "- dsl_* primitives (see dsl_primitives.py)"
    return (
        "You are a precise code generator. Output ONLY valid Python source code—no explanations, "
        "no markdown code fences.\n"
        "You must produce a single self-contained module that:\n"
        f"  1) begins with:  from {dsl_module} import *\n"
        "  2) defines:      def transform(grid: List[List[int]]) -> List[List[int]]\n"
        "  3) uses ONLY DSL primitives (functions whose names start with dsl_) to manipulate grids.\n"
        "\n"
        "STRICT RULES (CRITICAL):\n"
        "- You may NOT compute sizes, shapes, masks, frames, quadrants, slices, or bounding boxes yourself.\n"
        "- You may NOT index into grids (no grid[r][c], no slicing, no numpy; avoid any Subscript usage).\n"
        "- You may NOT introduce new arrays except those returned by DSL ops.\n"
        "- Control flow (for/if/while) is allowed to orchestrate DSL calls, but all grid manipulation must be via DSL ops.\n"
        "- If you need a value, region, or mask, obtain it with a DSL op only.\n"
        "\n"
        "AVAILABLE DSL PRIMITIVES (subset):\n"
        f"{op_list}\n"
        "\n"
        "Return ONLY Python source code for the module."
    )

def build_user_prompt(task_id: str, cand_id: str, instructions: List[str], train_shots: List[dict], feedback_text: str = "") -> str:
    instr_block = "\n".join(f"- {ln}" for ln in instructions) if instructions else "- <no instructions>"
    ex_json = ""
    if train_shots:
        ex_json = json.dumps(
            [{"input": ex["input"], "output": ex["output"]} for ex in train_shots if "input" in ex and "output" in ex],
            ensure_ascii=False
        )
    fb = ""
    if feedback_text.strip():
        fb = "\n\nPrevious errors to avoid (critical):\n" + feedback_text.strip() + "\n"

    return (
        f"Task ID: {task_id} | Candidate: {cand_id}\n"
        "Write Python that performs a grid transformation **using only the DSL primitives**. "
        "The module must start with from dsl_primitives import * and define:\n"
        " def transform(grid: List[List[int]]) -> List[List[int]]\n"
        "Natural-language steps (apply in order):\n"
        f"{instr_block}\n\n"
        + (("Training examples the code MUST satisfy (JSON list of {input, output}):\n" + ex_json + "\n\n") if ex_json else "")
        + fb +
        "Generate ONLY the Python source code."
    )


# ------------------------ HTTP to vLLM ------------------------

def post_chat_with_retries(url: str, headers: Dict[str,str], payload: Dict[str,Any],
                           retries: int, base_timeout: float) -> Tuple[int, Any, float, Optional[str]]:
    backoff = 0.8
    last_err = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=base_timeout)
            latency = round(time.time() - t0, 3)
            if r.status_code == 200:
                try:
                    return r.status_code, r.json(), latency, None
                except Exception as e:
                    return r.status_code, r.text, latency, f"json_decode_error: {e}"
            body_head = (r.text or "")[:400]
            last_err = f"HTTP {r.status_code}: {body_head}"
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                ra = r.headers.get("Retry-After")
                if ra:
                    try: sleep_s = float(ra)
                    except Exception: sleep_s = backoff * attempt
                else:
                    sleep_s = backoff * attempt
                time.sleep(sleep_s + random.random() * 0.3)
                continue
        except Exception as e:
            latency = round(time.time() - t0, 3)
            last_err = repr(e)
            if attempt < retries:
                time.sleep(backoff * attempt + random.random() * 0.3)
                continue
        return (r.status_code if 'r' in locals() else 0), (r.text if 'r' in locals() else None), latency, last_err
    return 0, None, 0.0, f"failed_after_retries: {last_err}"

def chat_complete_parallel(api_base: str, api_key: str, model: str,
    system_prompt: str, user_prompt: str,
    temperature: float, max_tokens: int,
    retries: int = 3, timeout: float = 120.0) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip() and api_key.strip().upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    status, body, latency, err = post_chat_with_retries(url, headers, payload, retries=retries, base_timeout=timeout)
    meta = {"latency_s": latency, "status_code": status}
    if isinstance(body, dict):
        meta["usage"] = body.get("usage")
        content = body.get("choices", [{}])[0].get("message", {}).get("content")
        return content, meta, err
    return None, meta, err

# ------------------------ extraction & repair ------------------------

def extract_python_source(text: str) -> str:
    if text is None: return ""
    t = text.strip()
    # Prefer fenced block if present
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", t, flags=re.IGNORECASE)
    if m: return m.group(1).strip()
    return t

def repair_source(src: str, dsl_module: str) -> str:
    if not src: return src

    # strip any stray triple backticks without language
    src = re.sub(r"^```+\s*$", "", src, flags=re.M)
    src = re.sub(r"^```+\s*$", "", src[::-1], flags=re.M)[::-1]

    # If it used alias import: "import dsl_primitives as dsl"
    alias_pat = rf"^\s*import\s+{re.escape(dsl_module)}\s+as\s+dsl\s*$"
    if re.search(alias_pat, src, flags=re.M):
        src = re.sub(alias_pat, "", src, flags=re.M)
        src = f"from {dsl_module} import *\n{src}"
        # Rewrite dsl.dsl_foo(...) → dsl_foo(...)
        src = re.sub(r"\bdsl\.(dsl_[A-Za-z0-9_]+)\b", r"\1", src)

    # If first non-empty line isn’t an import, prepend the DSL import
    lines = [ln for ln in src.splitlines() if ln.strip()]
    if not lines or not lines[0].lstrip().startswith(("from ", "import ")):
        src = f"from {dsl_module} import *\n\n{src}"

    return src

# ------------------------ validation ------------------------

def validate_python_module_basic(src: str) -> Optional[str]:
    try:
        ast.parse(src)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    tree = ast.parse(src)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        return "No function named 'transform' found."
    return None

def validate_dsl_only(src: str, dsl_module: str, allow_calls: Set[str], strict: bool) -> Optional[str]:
    """
    Relaxed by default: requires a DSL import, a transform(), and at least one dsl_* call.
    In --strict-dsl mode: also bans direct Subscript and non-DSL calls inside transform.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    # 1) must import DSL somewhere
    has_dsl_import = False
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == dsl_module:
            has_dsl_import = True; break
    if not has_dsl_import:
        return f"Missing 'from {dsl_module} import *' (or equivalent) at top-level."

    # 2) find transform
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transform"), None)
    if fn is None:
        return "Missing transform() definition."

    # 3) count dsl calls & optionally enforce strictness
    dsl_calls = 0

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            nonlocal dsl_calls
            fnname = None
            if isinstance(node.func, ast.Name):
                fnname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # accept attribute if attr starts with dsl_
                if getattr(node.func, "attr", "").startswith("dsl_"):
                    fnname = node.func.attr
            if fnname and fnname.startswith("dsl_"):
                dsl_calls += 1
            elif fnname and fnname in allow_calls:
                pass
            elif strict:
                raise RuntimeError(f"Call to non-DSL function '{fnname}' is not allowed in --strict-dsl mode.")
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript):
            if strict:
                raise RuntimeError("Forbidden direct indexing/slicing in --strict-dsl mode; use DSL ops only.")
            self.generic_visit(node)

    try:
        V().visit(fn)
    except RuntimeError as e:
        return str(e)

    if dsl_calls == 0:
        return "No DSL calls found; program must use at least one dsl_* primitive."
    return None

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--instr-json", required=True)
    ap.add_argument("--arc-root", default=None)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE","http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key",  default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--model",    default=os.environ.get("LLM_MODEL","llama-3.1-8b-instruct"))
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-tokens",  type=int, default=1600)
    ap.add_argument("--retries",     type=int, default=3)
    ap.add_argument("--timeout",     type=float, default=120.0)

    # Use a module (e.g., dsl_primitives) that exposes dsl_* functions
    ap.add_argument("--dsl-module",  default="dsl_primitives",
                    help="Python module that exposes dsl_* primitives (e.g., dsl_primitives)")

    # Strictness
    ap.add_argument("--strict-dsl", action="store_true",
                    help="Fail candidates that use Subscript or non-DSL calls inside transform(). Default: relaxed.")

    # Parallelism
    default_workers = min(max(6, (os.cpu_count() or 4) * 2), 16)
    ap.add_argument("--workers", type=int, default=default_workers)

    # Telemetry/printing
    ap.add_argument("--emit-events", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print-full-module", action="store_true",
                    help="Print full module (default prints only transform()).")
    ap.add_argument("--no-print", action="store_true", help="Don’t print any code.")
    # in main(), after other argparse adds:
    ap.add_argument("--feedback-file", default=None,
                    help="Optional path to a text file of prior errors/hints to avoid; added to the user prompt")


    args = ap.parse_args()

    task_dir = os.path.join(args.scores_root, f"{args.task_id}_allc_out")
    os.makedirs(task_dir, exist_ok=True)

    # Discover DSL ops to show the model
    dsl_ops = discover_dsl_ops(args.dsl_module)
    sys_prompt = build_system_prompt(args.dsl_module, dsl_ops)

    tel = Telemetry(scores_root=args.scores_root, emit_events=args.emit_events)
    tel.rollup.update({
        "task_id": args.task_id, "api_base": args.api_base, "model": args.model,
        "temperature": args.temperature, "max_tokens": args.max_tokens, "workers": args.workers,
    })

    # Preflight /models (warn-only)
    try:
        import urllib.request
        models_url = args.api_base.rstrip("/") + "/models"
        req = urllib.request.Request(models_url)
        if args.api_key and args.api_key.strip():
            req.add_header("Authorization", f"Bearer {args.api_key.strip()}")
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        avail = {m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)}
        if avail and args.model not in avail:
            log("WARN", f"Model '{args.model}' not served at {models_url}. Available: {sorted(list(avail))[:8]}…")
    except Exception as e:
        log("WARN", f"Preflight /models failed: {e} (continuing)")

    try:
        candidates = enumerate_candidates(args.instr_json)
    except Exception as e:
        log("ERROR", f"Failed to enumerate candidates: {e}")
        tel.rollup["notes"].append(f"enumerate_error:{e}")
        tel.end(); sys.exit(1)
    if not candidates:
        log("ERROR", "No instruction candidates found.")
        tel.rollup["notes"].append("no_candidates")
        tel.end(); sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        candidates = sorted(candidates, key=lambda x: x[0])

    tel.rollup["candidates_total"] = len(candidates)
    log("INFO", f"Task {args.task_id}: {len(candidates)} candidate(s); workers={args.workers}")

    train_shots: List[dict] = []
    if args.arc_root:
        try:
            task_obj = load_arc_task(args.arc_root, args.task_id)
            train_shots = pick_train_shots(task_obj, args.shots, args.seed) if args.shots else []
            log("INFO", f"Using {len(train_shots)} ARC training shot(s) in prompt.")
        except Exception as e:
            log("WARN", f"ARC examples unavailable: {e}")

    print_lock = threading.Lock()

    def process_candidate(cand: Tuple[str,int,List[str]]) -> Dict[str, Any]:
        cand_id, cand_idx, instructions = cand
        cand_key = f"{cand_id}" if cand_id is not None else str(cand_idx)
        out_py = os.path.join(task_dir, f"{args.task_id}.cand_{cand_key}.transform.nl.py")
        tel_json = os.path.join(task_dir, f"{args.task_id}.cand_{cand_key}.nl_gen.telemetry.json")
        per_tel: Dict[str, Any] = {
            "task_id": args.task_id, "candidate_id": cand_key, "candidate_index": cand_idx,
            "instr_json": args.instr_json, "out_py": out_py,
            "status": "init", "latency_s": None, "llm": {}, "prompt_bytes": {}, "generated_code": None,
        }
        try:
            feedback_text = ""
            if getattr(args, "feedback_file", None):
                try:
                    with open(args.feedback_file, "r", encoding="utf-8") as _fb:
                        feedback_text = _fb.read()
                except Exception:
                    feedback_text = ""

            user_prompt = build_user_prompt(
                args.task_id,
                cand_key,
                instructions,
                train_shots,
                feedback_text,  # <-- new arg
            )
            per_tel["prompt_bytes"] = {
                "system": len(sys_prompt.encode("utf-8")),
                "user": len(user_prompt.encode("utf-8")),
                "n_instructions": len(instructions),
                "shots": len(train_shots),
            }
            tel.event("llm.request", task_id=args.task_id, candidate=cand_key)
            content, meta, err = chat_complete_parallel(
                api_base=args.api_base, api_key=args.api_key, model=args.model,
                system_prompt=sys_prompt, user_prompt=user_prompt,
                temperature=args.temperature, max_tokens=args.max_tokens,
                retries=args.retries, timeout=args.timeout,
            )
            per_tel["llm"] = meta
            per_tel["latency_s"] = meta.get("latency_s")
            src_code = extract_python_source(content or "")
            src_code = repair_source(src_code, args.dsl_module)
            per_tel["generated_code"] = src_code

            # Validation
            v_err = validate_python_module_basic(src_code)
            if not v_err:
                v_err = validate_dsl_only(src_code, dsl_module=args.dsl_module,
                                          allow_calls=SAFE_BUILTIN_CALLS, strict=args.strict_dsl)

            if v_err:
                per_tel["status"] = "invalid_code"; per_tel["error"] = v_err
                tel.event("candidate.error", task_id=args.task_id, candidate=cand_key, error=v_err)
                # save sidecar for debugging
                try:
                    with open(out_py.replace(".py", ".invalid.txt"), "w", encoding="utf-8") as fe:
                        fe.write(v_err + "\n\n" + src_code)
                except Exception:
                    pass
            else:
                with open(out_py, "w", encoding="utf-8") as f:
                    f.write(src_code)
                per_tel["status"] = "ok"
                tel.event("candidate.ok", task_id=args.task_id, candidate=cand_key, out_py=out_py)

            # Print a banner per candidate unless --no-print
            if not args.no_print:
                banner = "=" * 80
                with print_lock:
                    print(f"\n{banner}\n[CODE] Task {args.task_id} | Candidate {cand_key} | status={per_tel['status']}\nPath: {out_py}\n{banner}", flush=True)
                    code_to_show = src_code or "# <empty model response>"
                    # Show only transform() for brevity
                    try:
                        tree = ast.parse(code_to_show)
                        fn = None
                        for n in tree.body:
                            if isinstance(n, ast.FunctionDef) and n.name == "transform":
                                fn = n; break
                        if fn and getattr(fn, "end_lineno", None) is not None:
                            lines = code_to_show.splitlines()
                            start = max(0, fn.lineno-1); end = min(len(lines), fn.end_lineno)
                            print("\n".join(lines[start:end]), flush=True)
                        else:
                            print(code_to_show, flush=True)
                    except Exception:
                        print(code_to_show, flush=True)
                    print(f"{banner}\n", flush=True)

            if args.verbose:
                log("INFO", f"candidate {cand_key} done: {per_tel.get('status')} {per_tel.get('error','')}")

        except Exception as e:
            per_tel["status"] = "exception"; per_tel["error"] = repr(e)
            tel.event("candidate.exception", task_id=args.task_id, candidate=cand_key, error=repr(e))
            if not args.no_print:
                with print_lock:
                    print(f"\n{'='*80}\n[CODE] Task {args.task_id} | Candidate {cand_key} | status=exception\n{e}\n{'='*80}\n", flush=True)
        try:
            with open(tel_json, "w", encoding="utf-8") as f:
                json.dump(per_tel, f, indent=2)
        except Exception:
            pass
        return per_tel

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_candidate, cand) for cand in candidates]
        for fut in as_completed(futs):
            per = fut.result()
            results.append(per)
            if per.get("status") == "ok":
                tel.rollup["candidates_ok"] += 1
            else:
                tel.rollup["candidates_err"] += 1

    tel.end()
    log("INFO", f"Telemetry summary: {tel.summary_path}")

if __name__ == "__main__":
    main()


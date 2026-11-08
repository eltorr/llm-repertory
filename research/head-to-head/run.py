"""Run the head-to-head (tasks.py) against the model the Repertory service serves now, and grade every answer.

    python research/head-to-head/run.py LABEL [--only calc,sched] [--url http://127.0.0.1:30000]

- Each task is one request; the tool task is a conversation that goes on while the model calls tools (the tools run
  against tasks.Desk, a simulated support backend).
- Every request: streamed, thinking on at reasoning effort "high" (what Hermes sends), the model's own sampling
  defaults (no temperature or top_p in the request), and no max_tokens. A task still running after LIMIT_S is cut
  off and fails.
- Code answers: the last ```python block, checked first (only the standard-library modules in SAFE_IMPORTS; no file,
  process or eval-style calls; no dunder tricks), then run with the task's tests in its own Python process
  (isolated mode, an empty folder, a timeout).
- Results: results/LABEL.json (per task: the grade, the answer, the reasoning, tokens, times; the setup it ran on,
  from the manager's status) and a table printed. `python -m repertory results add` logs them for docs/benchmarks.md.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks  # noqa: E402

LIMIT_S = 20 * 60
CODE_TIMEOUT_S = 60
SAFE_IMPORTS = {"time", "heapq", "collections", "functools", "itertools", "math", "re", "dataclasses", "typing",
                "bisect", "string", "operator", "enum", "copy"}
UNSAFE_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars",
                "memoryview", "help", "exit", "quit"}
SAFE_DUNDERS = {"__init__", "__call__", "__repr__", "__str__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__",
                "__ge__", "__hash__", "__len__", "__iter__", "__next__", "__contains__", "__getitem__",
                "__setitem__", "__delitem__", "__post_init__", "__slots__", "__name__", "__bool__"}
BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


# ----------------------------------------------------------------------------------------------------- requests
def stream_chat(url, body, deadline):
    """One streamed Chat Completions request: content, reasoning, tool calls (merged from their pieces), usage,
    time to the first token and in all. Cut off (timed_out) at `deadline` (time.monotonic)."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    out = {"content": "", "reasoning": "", "tool_calls": [], "usage": None, "finish": None, "ttft_s": None,
           "timed_out": False}
    calls = {}
    with urllib.request.urlopen(req, timeout=max(30.0, deadline - t0)) as r:
        for raw in r:
            if time.monotonic() > deadline:
                out["timed_out"] = True
                break                                   # closing the stream ends the request
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece and out["ttft_s"] is None:
                    out["ttft_s"] = round(time.monotonic() - t0, 2)
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or d.get("reasoning") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                out["finish"] = ch.get("finish_reason") or out["finish"]
    out["tool_calls"] = [calls[k] for k in sorted(calls)]
    out["total_s"] = round(time.monotonic() - t0, 2)
    return out


def body_for(messages, tools=None):
    b = {"model": "repertory", "messages": messages, "stream": True, "stream_options": {"include_usage": True},
         "reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}}
    if tools:
        b["tools"] = tools
    return b


# ------------------------------------------------------------------------------------------------- code checks
def unsafe(code):
    """Why the code may not run, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [n for n in names if n.split(".")[0] not in SAFE_IMPORTS]
            if bad:
                return f"imports {', '.join(bad)}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in UNSAFE_CALLS:
            return f"calls {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in SAFE_DUNDERS:
            return f"uses {node.attr}"
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in SAFE_DUNDERS:
            return f"uses {node.id}"
    return None


def grade_code(task, answer):
    """(passed all, passed, total, detail) for a code task's answer."""
    blocks = BLOCK.findall(answer or "")
    if not blocks:
        return False, 0, 0, "no python block"
    code = blocks[-1]
    why = unsafe(code)
    if why:
        return False, 0, 0, f"not run: {why}"
    with tempfile.TemporaryDirectory(prefix="h2h-") as d:
        p = Path(d) / "check.py"
        p.write_text(code + "\n\n" + task["tests"], encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP")}
        try:
            r = subprocess.run([sys.executable, "-I", "-S", str(p)], cwd=d, env=env, capture_output=True, text=True,
                               timeout=CODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return False, 0, 0, f"tests ran past {CODE_TIMEOUT_S} s"
    m = re.search(r"RESULT (\d+) (\d+) (.*)", r.stdout)
    if not m:
        return False, 0, 0, "the code failed before its tests: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    passed, total = int(m.group(1)), int(m.group(2))
    return passed == total, passed, total, ("all passed" if passed == total else f"failed: {m.group(3)[:300]}")


def manager_setup(url):
    """What the model ran with and on, from the Repertory manager's status (the dashboard token from its settings):
    the model's quant, tier, context and RAM cache, the GPU and the PC's RAM. None when the manager can't say."""
    try:
        sys.path.insert(0, str(HERE.parent.parent))
        from repertory import config
        req = urllib.request.Request(url + "/_switcher/api/status", headers={
            "Cookie": "repertory_token=" + config.load_config()["dashboard_token"]})
        s = json.load(urllib.request.urlopen(req, timeout=10))
        return {"quant": s["quant"], "tier": s["tier"], "context": s["context"], "ram_cache_gib": s["ram_cache_gib"],
                "vram_budget_gb": s["budget_gb"], "gpu": s["gpu"]["name"], "card_gb": round(s["gpu"]["total"] / 1e9, 2),
                "ram_gb": round(s["ram"]["total"] / 1e9, 1), "platform": s["platform"]}
    except Exception:  # noqa: BLE001 - the results stand without it
        return None


# ------------------------------------------------------------------------------------------------------- tasks
def run_task(url, task):
    deadline = time.monotonic() + LIMIT_S
    rec = {"id": task["id"], "name": task["name"], "requests": 0, "completion_tokens": 0, "prompt_tokens": 0,
           "seconds": 0.0, "timed_out": False}

    def ask(messages, tools=None):
        r = stream_chat(url, body_for(messages, tools), deadline)
        rec["requests"] += 1
        rec["seconds"] = round(rec["seconds"] + r["total_s"], 2)
        u = r["usage"] or {}
        rec["completion_tokens"] += u.get("completion_tokens") or 0
        rec["prompt_tokens"] += u.get("prompt_tokens") or 0
        rec["timed_out"] = rec["timed_out"] or r["timed_out"]
        rec.setdefault("ttft_s", r["ttft_s"])
        return r

    if task["kind"] == "tools":
        desk, messages, final = tasks.Desk(), [{"role": "user", "content": task["prompt"]}], None
        rec["transcript"] = []
        for _ in range(16):
            r = ask(messages, task["tools"])
            rec["transcript"].append({"content": r["content"], "reasoning_chars": len(r["reasoning"]),
                                      "tool_calls": r["tool_calls"]})
            if r["timed_out"]:
                break
            if not r["tool_calls"]:
                final = r["content"]
                break
            calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                      "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                     for i, c in enumerate(r["tool_calls"])]
            messages.append({"role": "assistant", "content": r["content"] or "", "tool_calls": calls})
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"])
                    result = desk.call(c["function"]["name"], args if isinstance(args, dict) else {})
                except ValueError:
                    result = {"error": "arguments are not valid JSON"}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
        checks = tasks.grade_ticket(desk, final)
        rec.update(ok=all(checks.values()) and not rec["timed_out"], score=f"{sum(checks.values())}/{len(checks)}",
                   detail="; ".join(f"{'ok' if v else 'FAIL'} {k}" for k, v in checks.items()),
                   calls=[(n, a) for n, a in desk.calls], answer=final)
        return rec
    r = ask([{"role": "user", "content": task["prompt"]}])
    rec.update(answer=r["content"], reasoning=r["reasoning"], finish=r["finish"])
    if r["timed_out"]:
        rec.update(ok=False, score="-", detail=f"cut off after {LIMIT_S // 60} min")
    elif task["kind"] == "code":
        ok, passed, total, detail = grade_code(task, r["content"])
        rec.update(ok=ok, score=f"{passed}/{total}", detail=detail)
    else:
        ok, detail = tasks.grade_answer_line(task["id"], r["content"])
        rec.update(ok=ok, score="1/1" if ok else "0/1", detail=detail)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("label", help="the results' name, e.g. qwen-4.05")
    p.add_argument("--only", help="task ids, comma-separated")
    p.add_argument("--url", default="http://127.0.0.1:30000")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[\w.-]+", a.label):
        p.error("the label may only have letters, digits, dots, dashes and underscores")
    chosen = [t for t in tasks.TASKS if not a.only or t["id"] in a.only.split(",")]
    models = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"]
    out = {"label": a.label, "model": models[0]["id"] if models else None,
           "date": time.strftime("%Y-%m-%d %H:%M"), "limit_s": LIMIT_S, "setup": manager_setup(a.url), "tasks": []}
    path = HERE / "results" / f"{a.label}.json"
    path.parent.mkdir(exist_ok=True)
    for t in chosen:
        print(f"== {t['id']}: {t['name']}", flush=True)
        try:
            rec = run_task(a.url, t)
        except Exception as e:  # noqa: BLE001 - one failed task must not end the run
            rec = {"id": t["id"], "name": t["name"], "ok": False, "score": "-", "detail": f"request failed: {e}"}
        out["tasks"].append(rec)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        tps = rec.get("completion_tokens", 0) / rec["seconds"] if rec.get("seconds") else 0
        print(f"   {'PASS' if rec['ok'] else 'FAIL'} {rec['score']}  {rec.get('completion_tokens', 0):,} tokens in "
              f"{rec.get('seconds', 0):.0f} s ({tps:.1f} tok/s)  {rec['detail'][:200]}", flush=True)
    end = manager_setup(a.url)
    if end and out["setup"] and any(end[k] != out["setup"][k] for k in ("quant", "tier", "context", "ram_cache_gib")):
        out["setup_changed"] = end                      # e.g. the manager went to lean for a game meanwhile
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("the model's setup changed during the run:", end)
    won = sum(1 for r in out["tasks"] if r["ok"])
    print(f"{a.label}: {won}/{len(out['tasks'])} passed; results in {path}")


if __name__ == "__main__":
    main()

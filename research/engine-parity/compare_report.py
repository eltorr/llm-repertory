"""Compare two engine_compare.py runs: identical answers, and the timings side by side."""
import difflib
import json
import sys

a = {r["name"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))["results"]}
b = {r["name"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))["results"]}
same_all = True
for name in a:
    x, y = a[name], b.get(name)
    if y is None:
        print(f"{name}: missing in the second run"); same_all = False; continue
    def calls(r):
        return [((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments"))
                for c in r["tool_calls"]]
    same = (x["content"] == y["content"] and x["reasoning"] == y["reasoning"] and calls(x) == calls(y)
            and x["finish"] == y["finish"])
    same_all &= same
    ux, uy = x.get("usage") or {}, y.get("usage") or {}
    print(f"{name:9s} {'IDENTICAL' if same else 'DIFFERENT'}  tokens {ux.get('completion_tokens')}/{uy.get('completion_tokens')}"
          f"  first token {x['ttft_s']}/{y['ttft_s']} s  decode {x['decode_tps']}/{y['decode_tps']} tok/s")
    if not same:
        for part in ("reasoning", "content"):
            if x[part] != y[part]:
                d = list(difflib.unified_diff(x[part].splitlines(), y[part].splitlines(), lineterm="", n=1))
                print(f"  {part} differs:"); print("   " + "\n   ".join(d[:20]))
        if calls(x) != calls(y):
            print("  tool calls:", calls(x), "vs", calls(y))
print("ALL IDENTICAL" if same_all else "SOME DIFFER")

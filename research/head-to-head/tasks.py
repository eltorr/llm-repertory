"""The head-to-head's tasks: short to run, each needing real reasoning, each graded by code (run.py runs them).

Every task is a dict: id, kind ("code", "answer" or "tools"), the user prompt, and a grader. Answers that can be
computed are computed here at import (the counting problem by DP, the puzzle by brute force, which also checks that
its solution is unique, the log's downtime by replaying it), so the expected answers can't drift from the tasks.
"""
import itertools
import random
import re
import shop

ANSWER = re.compile(r"ANSWER:\s*(.+)", re.I)


def final_answer(text):
    """The last 'ANSWER: ...' line of an answer, stripped; None without one."""
    found = ANSWER.findall(text or "")
    return found[-1].strip().strip("*`").strip() if found else None


# --------------------------------------------------------------------------------------- 1. expression evaluator
CALC_PROMPT = """Write a Python function `evaluate(expr: str) -> int` that evaluates an integer arithmetic expression.

- Tokens: non-negative integer literals (any number of digits, leading zeros allowed), the binary operators + - * / ^,
  unary minus, parentheses, and any whitespace between tokens.
- Precedence, lowest first: + and - (left-associative); * and / (left-associative); unary minus; ^ (right-associative,
  binding tighter than unary minus, so -2^2 is -4). The right operand of ^ may itself start with unary minus.
- / is integer division truncating toward zero (-7 / 2 is -3). Division by zero raises ZeroDivisionError.
- A negative exponent raises ValueError. Any malformed input raises ValueError: empty input, unbalanced parentheses,
  a missing operand, two numbers in a row, an unknown character, an empty pair of parentheses.

Use only the standard library and no imports. Reply with the function in a single ```python block."""

CALC_CASES = [
    ("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("10 / 3", 3), ("-7 / 2", -3), ("7 / -2", -3), ("-2^2", -4),
    ("(-2)^2", 4), ("2^3^2", 512), ("2 ^ 10 - 1", 1023), ("--3", 3), ("3 - -3", 6), ("2*-3", -6), ("  42  ", 42),
    ("100 - 10 - 1", 89), ("64 / 4 / 2", 8), ("007 + 1", 8), ("-(3 + 4) * 2", -14), ("2 * (3 + (4 - 1)) ^ 2", 72),
    ("1 - 2^2^0", -1), ("-3^2^0", -3), ("12345678901234567890 * 0 + 5", 5), ("2^-0", 1),
    ("", ValueError), ("   ", ValueError), ("1 +", ValueError), ("* 2", ValueError), ("(1 + 2", ValueError),
    ("1 + 2)", ValueError), ("1 2", ValueError), ("2 ^ -1", ValueError), ("3 $ 4", ValueError), ("()", ValueError),
    ("1 / 0", ZeroDivisionError), ("5 / (2 - 2)", ZeroDivisionError),
]

CALC_TESTS = """
cases = %r
errors = {"ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError}
passed, failed = 0, []
for expr, want in cases:
    try:
        got = evaluate(expr)
        ok = isinstance(want, int) and got == want and type(got) is int
    except Exception as e:  # noqa: BLE001
        got, ok = type(e).__name__, isinstance(want, str) and type(e).__name__ == want
    passed += ok
    if not ok:
        failed.append((expr, want, got))
print("RESULT", passed, len(cases), repr(failed[:6]))
""" % ([(e, w if isinstance(w, int) else w.__name__) for e, w in CALC_CASES],)


# ----------------------------------------------------------------------------------------- 2. task scheduler
SCHED_PROMPT = """Write a Python function `schedule(durations, deps, workers)` that simulates this scheduler and
returns `(makespan, starts)`.

- `durations`: dict task name -> positive int duration. `deps`: dict task name -> list of task names it depends on
  (a task missing from `deps` has none). `workers`: number of identical workers (>= 1).
- Time starts at 0. A task is ready once every task it depends on has finished; a task finishing at time t makes its
  dependents ready at time t.
- At any time t: first every task finishing at t finishes; then, while a worker is idle and a task is ready, the idle
  worker starts the ready task with the longest duration, ties broken by the smaller name (string order).
- Return the time the last task finishes (`makespan`, 0 when there are no tasks) and `starts`: dict task -> start time.
- Raise ValueError if `deps` names a task that isn't in `durations`, or if the dependencies contain a cycle.

Use only the standard library (heapq and collections are fine). Reply with the function in a single ```python block."""


def schedule_reference(durations, deps, workers):
    for t, ds in deps.items():
        if t not in durations or any(d not in durations for d in ds):
            raise ValueError("unknown task")
    need = {t: set(deps.get(t, ())) for t in durations}
    users = {t: [] for t in durations}
    for t, ds in need.items():
        for d in ds:
            users[d].append(t)
    done, starts, running, now, idle = set(), {}, [], 0, workers
    ready = {t for t, ds in need.items() if not ds}
    while len(done) < len(durations):
        while idle and ready:
            t = min(ready, key=lambda x: (-durations[x], x))
            ready.discard(t)
            starts[t] = now
            running.append((now + durations[t], t))
            idle -= 1
        if not running:
            raise ValueError("cycle")
        now = min(end for end, _ in running)
        for end, t in [r for r in running if r[0] == now]:
            running.remove((end, t))
            done.add(t)
            idle += 1
            for u in users[t]:
                need[u].discard(t)
                if not need[u] and u not in done and u not in starts:
                    ready.add(u)
    return (max((starts[t] + durations[t] for t in durations), default=0), starts)


def schedule_cases():
    rng = random.Random(7)
    cases = [({}, {}, 2), ({"a": 3}, {}, 1), ({"a": 2, "b": 2, "c": 1}, {"c": ["a", "b"]}, 1),
             ({"x": 4, "y": 4, "z": 4}, {}, 2), ({"p": 1, "q": 5, "r": 5}, {"r": ["p"]}, 2)]
    for n, w in ((8, 2), (10, 3), (12, 2), (14, 4), (16, 3), (9, 1)):
        names = [f"t{i:02d}" for i in range(n)]
        durations = {t: rng.choice([1, 2, 2, 3, 3, 4, 5, 6]) for t in names}
        deps = {t: sorted(rng.sample(names[:i], k=min(i, rng.choice([0, 1, 1, 2, 3])))) for i, t in enumerate(names)}
        cases.append((durations, {t: d for t, d in deps.items() if d}, w))
    out = [(d, p, w, schedule_reference(d, p, w)) for d, p, w in cases]
    out.append(({"a": 1, "b": 1}, {"a": ["b"], "b": ["a"]}, 2, "ValueError"))
    out.append(({"a": 1}, {"a": ["ghost"]}, 1, "ValueError"))
    return out


SCHED_TESTS = """
cases = %r
passed, failed = 0, []
for durations, deps, workers, want in cases:
    try:
        got = schedule(durations, deps, workers)
        got = (got[0], dict(got[1]))
        ok = want != "ValueError" and tuple(got) == tuple(want)
    except Exception as e:  # noqa: BLE001
        got, ok = type(e).__name__, want == "ValueError" and type(e).__name__ == "ValueError"
    passed += ok
    if not ok:
        failed.append((sorted(durations.items())[:4], workers, str(want)[:120], str(got)[:120]))
print("RESULT", passed, len(cases), repr(failed[:3]))
""" % (schedule_cases(),)


# -------------------------------------------------------------------------------------------- 3. the bug fix
CACHE_PROMPT = '''This class has several bugs. Find them and reply with the corrected class, complete and with the same
interface, in a single ```python block. Keep `import time` (the default clock); use no other imports.

```python
import time


class TTLCache:
    """A least-recently-used cache whose entries also expire. An entry written at time t is live while clock() < t + ttl;
    after that it is gone: never returned, and it never counts toward capacity. get(key) returns the value of a live
    entry (a use) or None. put(key, value) writes the entry (a use, and its write time is now). When a put leaves more
    than capacity live entries, the least recently used live entry is evicted. clock returns the time in seconds."""

    def __init__(self, capacity, ttl, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.data = {}          # key -> (value, written_at)
        self.order = []         # keys, least recently used first

    def _expired(self, key):
        return self.clock() - self.data[key][1] > self.ttl

    def get(self, key):
        if key not in self.data:
            return None
        if self._expired(key):
            return None
        self.order.remove(key)
        self.order.insert(0, key)
        return self.data[key][0]

    def put(self, key, value):
        if key in self.data:
            self.order.remove(key)
        self.data[key] = (value, self.clock())
        self.order.append(key)
        if len(self.data) > self.capacity:
            oldest = self.order.pop()
            del self.data[oldest]
```'''

CACHE_TESTS = """
class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

c = Clock(); x = TTLCache(2, 10, clock=c)
x.put("a", 1); x.put("b", 2); check("basic", x.get("a") == 1 and x.get("b") == 2)

c = Clock(); x = TTLCache(2, 100, clock=c)
x.put("a", 1); x.put("b", 2); x.get("a"); x.put("c", 3)
check("get refreshes recency: b evicted", x.get("b") is None and x.get("a") == 1 and x.get("c") == 3)

c = Clock(); x = TTLCache(2, 100, clock=c)
x.put("a", 1); x.put("b", 2); x.put("c", 3)
check("evicts the least recently used", x.get("a") is None and x.get("b") == 2 and x.get("c") == 3)

c = Clock(); x = TTLCache(3, 5, clock=c)
x.put("a", 1); c.t = 4.999; check("live just before ttl", x.get("a") == 1)
c.t = 5.0; check("gone at exactly ttl", x.get("a") is None)

c = Clock(); x = TTLCache(2, 5, clock=c)
x.put("a", 1); x.put("b", 2); c.t = 10; x.put("c", 3); x.put("d", 4)
check("expired entries don't count toward capacity", x.get("c") == 3 and x.get("d") == 4)

c = Clock(); x = TTLCache(2, 5, clock=c)
x.put("a", 1); c.t = 3; x.put("b", 2); c.t = 6; x.put("c", 3)
check("expired a doesn't push out live b", x.get("b") == 2 and x.get("c") == 3 and x.get("a") is None)

c = Clock(); x = TTLCache(2, 5, clock=c)
x.put("a", 1); c.t = 4; x.put("a", 9); c.t = 8
check("a rewrite renews the write time", x.get("a") == 9)

c = Clock(); x = TTLCache(2, 100, clock=c)
x.put("a", 1); x.put("b", 2); x.put("a", 3); x.put("c", 4)
check("a rewrite is a use: b evicted", x.get("b") is None and x.get("a") == 3 and x.get("c") == 4)

c = Clock(); x = TTLCache(1, 100, clock=c)
x.put("a", 1); x.put("b", 2); check("capacity one", x.get("a") is None and x.get("b") == 2)

c = Clock(); x = TTLCache(2, 5, clock=c)
x.put("a", 1); c.t = 6; x.get("a"); x.put("b", 2); x.put("c", 3)
check("an expired get leaves no ghost behind", x.get("b") == 2 and x.get("c") == 3)

print("RESULT", sum(ok for _, ok in checks), len(checks), repr([n for n, ok in checks if not ok]))
"""


# ---------------------------------------------------------------------------------------------- 4. counting
PATHS_PROMPT = """A path on the grid goes from (0, 0) to (10, 10) in unit steps, each step either right (x + 1) or up
(y + 1). How many such paths never visit a point (k, k) where k is odd? (The start and end points have even k.)

Work it out by reasoning; you can't run code. End with a line `ANSWER: <the number>`."""


def paths_reference(n=10):
    grid = [[0] * (n + 1) for _ in range(n + 1)]
    for x in range(n + 1):
        for y in range(n + 1):
            if x == y and x % 2 == 1:
                continue
            grid[x][y] = 1 if x == y == 0 else (grid[x - 1][y] if x else 0) + (grid[x][y - 1] if y else 0)
    return grid[n][n]


PATHS_ANSWER = paths_reference()


# ----------------------------------------------------------------------------------------- 5. logic puzzle
PEOPLE = ("Ana", "Ben", "Cara", "Dev", "Eli")
LANGS = ("Python", "Rust", "Go", "Java", "C")
CLUES = [
    "Each of the five works on a different floor, 1 to 5, and each uses a different one of the five languages.",
    "1. The person on floor 1 uses C.",
    "2. Ben works on the floor directly above the Go programmer.",
    "3. The Java programmer works on the floor directly below the Python programmer.",
    "4. Cara uses neither Python nor Java, and works on a lower floor than Ben.",
    "5. Dev works on an even-numbered floor.",
    "6. Eli works on a higher floor than Dev.",
    "7. The Rust programmer works on a higher floor than Ana.",
    "8. Ana does not work on floor 1.",
    "9. Dev works on a higher floor than Ana.",
    "10. Ben does not use Java.",
]


def puzzle_solutions():
    sols = []
    for floors in itertools.permutations(range(1, 6)):
        f = dict(zip(PEOPLE, floors))
        for langs in itertools.permutations(LANGS):
            lang = dict(zip(PEOPLE, langs))
            who = {lang[p]: p for p in PEOPLE}
            on = {f[p]: p for p in PEOPLE}
            if (lang[on[1]] == "C" and f["Ben"] == f[who["Go"]] + 1 and f[who["Java"]] + 1 == f[who["Python"]]
                    and lang["Cara"] not in ("Python", "Java") and f["Cara"] < f["Ben"] and f["Dev"] % 2 == 0
                    and f["Eli"] > f["Dev"] and f[who["Rust"]] > f["Ana"] and f["Ana"] != 1
                    and f["Dev"] > f["Ana"] and lang["Ben"] != "Java"):
                sols.append((f, lang))
    return sols


_SOLS = puzzle_solutions()
assert len(_SOLS) == 1, f"the puzzle must have one solution, it has {len(_SOLS)}"
_F, _L = _SOLS[0]
PUZZLE_ANSWER = next(p for p in PEOPLE if _F[p] == 3), _L[next(p for p in PEOPLE if _F[p] == 3)]
PUZZLE_PROMPT = ("Five developers, " + ", ".join(PEOPLE) + ", and five languages, " + ", ".join(LANGS) + ".\n\n"
                 + "\n".join(CLUES) + "\n\nWho works on floor 3, and which language do they use? "
                 "End with a line `ANSWER: <name>, <language>`.")


# ------------------------------------------------------------------------------------- 6. tools: a support ticket
CUSTOMERS = {"maria.lopez@example.com": {"customer_id": "C101", "name": "Maria Lopez"},
             "maria.lopes@example.com": {"customer_id": "C102", "name": "Maria Lopes"},
             "m.lopez@example.com": {"customer_id": "C103", "name": "Miguel Lopez"}}
ORDERS = {
    "O-5001": {"customer_id": "C101", "date": "2026-03-04", "items": [
        {"sku": "MUG-12", "name": "Ceramic mug", "qty": 2, "price": 12.0},
        {"sku": "TEA-01", "name": "Green tea 100g", "qty": 1, "price": 9.5}]},
    "O-5002": {"customer_id": "C101", "date": "2026-03-19", "items": [
        {"sku": "GLS-08", "name": "Glass carafe 1L", "qty": 1, "price": 34.0},
        {"sku": "CST-04", "name": "Cork coasters (4)", "qty": 1, "price": 8.0}]},
    "O-5003": {"customer_id": "C101", "date": "2026-02-27", "items": [
        {"sku": "GLS-02", "name": "Glass tumblers (2)", "qty": 1, "price": 18.0}]},
    "O-5004": {"customer_id": "C102", "date": "2026-03-20", "items": [
        {"sku": "GLS-08", "name": "Glass carafe 1L", "qty": 1, "price": 34.0}]},
}
TOOLS = [
    {"type": "function", "function": {"name": "find_customer", "description": "Look up a customer by exact email.",
     "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}},
    {"type": "function", "function": {"name": "list_orders",
     "description": "A customer's orders in one month: order id, date and total.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"},
                                                     "month": {"type": "string", "description": "YYYY-MM"}},
                    "required": ["customer_id", "month"]}}},
    {"type": "function", "function": {"name": "get_order", "description": "An order's items (sku, name, qty, price).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "refund_item",
     "description": "Refund some quantity of one item of an order. Returns the refund id and amount.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "sku": {"type": "string"},
                                                     "quantity": {"type": "integer"}, "reason": {"type": "string"}},
                    "required": ["order_id", "sku", "quantity", "reason"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Email a customer.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                                                     "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]}}},
]
TICKET_PROMPT = """You are a support agent with tools. Today is 2026-04-02.

Ticket from maria.lopez@example.com: "The glass thing from my order last month arrived cracked. Please refund just
that item."

Handle it with the tools: refund only the damaged item, then email the customer a confirmation that includes the refund
id and the amount. When done, reply with a one-line summary."""


class Desk:
    """The tools' simulated backend; it records every call."""

    def __init__(self):
        self.calls, self.refunds, self.emails = [], [], []

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "find_customer":
            c = CUSTOMERS.get(str(args.get("email", "")).strip().lower())
            return c or {"error": "no customer with that email"}
        if name == "list_orders":
            cid, month = args.get("customer_id"), str(args.get("month", ""))
            return [{"order_id": o, "date": v["date"], "total": sum(i["qty"] * i["price"] for i in v["items"])}
                    for o, v in ORDERS.items() if v["customer_id"] == cid and v["date"].startswith(month)]
        if name == "get_order":
            o = ORDERS.get(args.get("order_id"))
            return dict(o, order_id=args.get("order_id")) if o else {"error": "no such order"}
        if name == "refund_item":
            o = ORDERS.get(args.get("order_id"))
            item = next((i for i in (o or {}).get("items", []) if i["sku"] == args.get("sku")), None)
            try:
                qty = int(args.get("quantity", 0))
            except (TypeError, ValueError):
                qty = 0
            if not item or not 0 < qty <= item["qty"]:
                return {"error": "no such item in that order, or a bad quantity"}
            rid = f"R-{7301 + len(self.refunds)}"
            self.refunds.append({"refund_id": rid, "order_id": args["order_id"], "sku": item["sku"], "quantity": qty,
                                 "amount": qty * item["price"]})
            return {"refund_id": rid, "amount": qty * item["price"]}
        if name == "send_email":
            self.emails.append(args)
            return {"sent": True}
        return {"error": f"unknown tool {name}"}


def grade_ticket(desk, final):
    right = [r for r in desk.refunds if r["order_id"] == "O-5002" and r["sku"] == "GLS-08" and r["quantity"] == 1]
    wrong = [r for r in desk.refunds if r not in right]
    mail = [e for e in desk.emails if str(e.get("to", "")).lower() == "maria.lopez@example.com"
            and right and right[0]["refund_id"] in str(e.get("body", "")) and "34" in str(e.get("body", ""))]
    checks = {"refunded the carafe from the March order": len(right) == 1, "no other refund": not wrong,
              "emailed the customer the refund id and amount": bool(mail),
              "emailed no one else": all(str(e.get("to", "")).lower() == "maria.lopez@example.com" for e in desk.emails),
              "replied with a summary": bool((final or "").strip())}
    return checks


# --------------------------------------------------------------------------------- 7. long context: the incident log
SERVICES = ("auth", "billing", "search", "media", "notify", "maps")
NOISE = ["INFO request latency p50={a}ms p99={b}ms", "INFO deploy {s}-{v} rolled out to {n} pods",
         "WARN latency above SLO for {a}s", "INFO cache hit ratio {c}%", "INFO autoscaler set replicas={n}",
         "WARN retrying upstream call ({n}/3)", "INFO health probe p{n} ok", "INFO config reload ({v})",
         "DEBUG gc pause {a}ms", "INFO queue depth {b}"]


def build_log():
    """(log text, {service: downtime seconds}): outages, the traps, and noise, in time order."""
    rng = random.Random(1717)
    events = []                                   # (seconds, service, text)

    def at(h, m, s=0):
        return h * 3600 + m * 60 + s

    def outage(svc, start, end, dup_every=0):
        events.append((start, svc, f"[{svc}] DOWN detected by probe p{rng.randint(1, 9)}"))
        t = start + dup_every
        while dup_every and t < end - 30:         # repeated DOWN reports while already down
            events.append((t, svc, f"[{svc}] DOWN still failing (probe p{rng.randint(1, 9)})"))
            t += dup_every
        events.append((end, svc, f"[{svc}] RECOVERED"))

    outage("search", at(2, 11, 4), at(2, 58, 49), dup_every=600)      # 47m45s
    outage("search", at(16, 40, 0), at(17, 26, 30), dup_every=420)    # 46m30s  -> 94m15s
    t = at(5, 0)
    for k in range(9):                                                # notify flaps: nine short outages
        length = (11 * 60 + 7) + (k % 3) * 37
        outage("notify", t, t + length)
        t += length + rng.randint(25, 70) * 60
    outage("billing", at(9, 2, 10), at(10, 30, 40), dup_every=900)    # 88m30s
    events.append((at(13, 0), "billing", "[billing] [test] DOWN synthetic alert, ignore"))
    events.append((at(14, 0), "billing", "[billing] [test] RECOVERED synthetic alert, ignore"))
    outage("maps", at(20, 15, 0), at(21, 21, 0))                      # 66m
    outage("auth", at(11, 5, 5), at(11, 49, 35), dup_every=300)       # 44m30s
    for _ in range(620):
        sec = rng.randint(0, 86399)
        svc = rng.choice(SERVICES)
        text = rng.choice(NOISE).format(a=rng.randint(5, 900), b=rng.randint(10, 4000), s=svc,
                                        v=f"{rng.randint(1, 9)}.{rng.randint(0, 40)}", n=rng.randint(1, 30),
                                        c=rng.randint(40, 99))
        events.append((sec, svc, f"[{svc}] {text}"))
    events.sort(key=lambda e: (e[0], e[2]))
    lines = [f"2026-06-14 {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d} {text}" for s, _, text in events]
    down, since = {}, {}
    for s, svc, text in events:                   # the rule the prompt states, replayed
        if "[test]" in text:
            continue
        if " DOWN" in text and svc not in since:
            since[svc] = s
        elif text.endswith("RECOVERED") and svc in since:
            down[svc] = down.get(svc, 0) + s - since.pop(svc)
    return "\n".join(lines), down


LOG_TEXT, LOG_DOWN = build_log()
LOG_TOP = max(LOG_DOWN, key=LOG_DOWN.get)
LOG_ANSWER = (LOG_TOP, LOG_DOWN[LOG_TOP] // 60)
LOG_PROMPT = f"""Below is a day of incident log for six services. A service is down from a line saying it is DOWN until
the next line saying it RECOVERED; further DOWN lines while it is already down don't restart the clock. Lines marked
[test] are synthetic alerts: ignore them.

Which service had the most total downtime that day, and how long was it in whole minutes (rounded down)?
End with a line `ANSWER: <service>, <minutes>`.

<log>
{LOG_TEXT}
</log>"""


def grade_answer_line(task_id, text):
    got = final_answer(text)
    if got is None:
        return False, "no ANSWER line"
    if task_id == "paths":
        nums = re.findall(r"-?\d[\d,]*", got)
        ok = bool(nums) and int(nums[0].replace(",", "")) == PATHS_ANSWER
        return ok, f"answered {got!r}, expected {PATHS_ANSWER}"
    if task_id == "puzzle":
        name, lang = PUZZLE_ANSWER
        ok = name.lower() in got.lower() and re.search(rf"\b{re.escape(lang.lower())}\b", got.lower()) is not None
        return ok, f"answered {got!r}, expected {name}, {lang}"
    if task_id == "log":
        svc, minutes = LOG_ANSWER
        nums = [int(n) for n in re.findall(r"\d+", got)]
        ok = svc in got.lower() and minutes in nums
        return ok, f"answered {got!r}, expected {svc}, {minutes}"
    raise KeyError(task_id)


TASKS = [
    {"id": "calc", "kind": "code", "name": "Expression evaluator", "prompt": CALC_PROMPT, "tests": CALC_TESTS},
    {"id": "sched", "kind": "code", "name": "Scheduler simulation", "prompt": SCHED_PROMPT, "tests": SCHED_TESTS},
    {"id": "cache", "kind": "code", "name": "Fix the TTL cache", "prompt": CACHE_PROMPT, "tests": CACHE_TESTS},
    {"id": "paths", "kind": "answer", "name": "Counting paths", "prompt": PATHS_PROMPT},
    {"id": "puzzle", "kind": "answer", "name": "Logic puzzle", "prompt": PUZZLE_PROMPT},
    {"id": "ticket", "kind": "tools", "name": "Support ticket with tools", "prompt": TICKET_PROMPT, "tools": TOOLS},
    {"id": "log", "kind": "answer", "name": "Incident log (long context)", "prompt": LOG_PROMPT},
    {"id": "shop", "kind": "session", "name": "Coding session: fix and extend a project", "prompt": shop.PROMPT,
     "tools": shop.TOOLS, "limit_s": 30 * 60},
]

if __name__ == "__main__":
    print("paths:", PATHS_ANSWER, "| puzzle:", PUZZLE_ANSWER, "| log:", LOG_ANSWER, sorted(LOG_DOWN.items()))
    print("log lines:", LOG_TEXT.count("\n") + 1, "chars:", len(LOG_TEXT))

# Head-to-head: Qwen3.8-Flash-Next and GLM-5.3-Flash

2026-11-08, RTX 5090 (32 GB) with 64 GB of RAM, Windows 11 and WSL2, the full profile at 256K of context.

Three setups that fit this PC, on seven short tasks that each need real reasoning: is GLM-5.3-Flash, at a quant this
PC runs at a usable speed, worth it for coding sessions?

| Setup | RAM cache | Decode (curation's requests) | Prefill (58-73K document) |
|---|---|---|---|
| Qwen3.8-Flash-Next 4.05 bpw | 35 GiB | 42.9 tok/s | 6,209 tok/s |
| GLM-5.3-Flash 3.05 bpw | 37 GiB | 8.8 tok/s | 1,228 tok/s |
| GLM-5.3-Flash 2.05 bpw | 40 GiB | 20.1 tok/s | 2,262 tok/s |

## The tasks

[tasks.py](tasks.py); [run.py](run.py) runs them against the model the service serves and grades every answer.

| Task | Kind | Graded by |
|---|---|---|
| calc | Code: an integer expression evaluator (precedence, right-associative `^`, unary minus, errors) | 34 cases |
| sched | Code: simulate a task scheduler with dependencies and N workers | 13 cases against a reference simulation |
| cache | Code: fix the five bugs in a TTL + LRU cache | 11 checks |
| paths | Counting lattice paths under a constraint | the exact count (computed by DP) |
| puzzle | A logic puzzle | the unique solution (checked by brute force) |
| ticket | A support ticket, solved with five simulated tools | 5 checks of the calls made and the reply |
| log | A day's incident log: the service with the most downtime | the service and its minutes (replayed) |

Every request is streamed with thinking on at reasoning effort `high` (what Hermes sends), the model's own sampling
values and no `max_tokens`. A task still running after 20 minutes is cut off and fails. Code runs in its own Python
process (isolated mode, an empty folder, a timeout) after a check that it imports only safe modules and calls no
file, process or eval-style functions.

`high` is not the same level for both models. Qwen's template has low, medium and xhigh, so `high` runs at xhigh,
its longest. GLM's has Low, High and Max, so `high` runs at High. Hermes gets the same mapping, so this is how the
models behave for it today.

## Results

| Task | Qwen 4.05 | GLM 3.05 | GLM 2.05 |
|---|---|---|---|
| calc | cut off at 20 min (looped) | pass, 1.7K tokens, 3.7 min | pass, 4.9K tokens, 3.9 min |
| sched | pass, 17.4K tokens, 6.5 min | fail 3/13 | pass, 8.5K tokens, 6.6 min |
| cache | pass, 41.6K tokens, 18.6 min | pass, 2.0K tokens, 4.2 min | fail 10/11, 1.8 min |
| paths | pass, 16.7K tokens, 6.3 min | cut off at 20 min | pass, 7.3K tokens, 5.7 min |
| puzzle | pass, 0.4 min | pass, 1.8 min | pass, 1.2 min |
| ticket | pass, 0.4 min | pass, 1.9 min | pass, 0.6 min |
| log | pass, 2.7 min | pass, 4.9 min | fail (answered search, 93) |
| **Passed** | **6/7** | **5/7** | **5/7** |
| **Coding tasks** | **2/3 in 45.2 min** | **2/3 in 11.3 min** | **2/3 in 12.3 min** |
| All seven | 55.0 min | 39.8 min | 22.0 min |

Token counts of a cut-off task aren't known: the stream is closed before its usage arrives.

What went wrong:

- **Qwen 4.05, calc:** it reasoned past 20 minutes without answering. On cache it got there, after 41.6K tokens.
- **GLM 3.05, sched:** a worker loop that stops while tasks are still running (`if not running` without the work
  left), so dependent tasks never start. **paths:** a sound approach that didn't finish in 20 minutes at 8 tok/s.
- **GLM 2.05, cache:** four of the five bugs fixed, but `get` moves the key to the front of the recency list instead
  of the back, so the cache evicts the entry just used. **log:** it rounded each outage down to whole minutes before
  adding them up (the task rounds the total), which made notify's 105 minutes 91 and gave the answer to search (93).

## What it says about coding sessions

- **The same accuracy on code.** Each setup solved two of the three coding tasks, a different two each time. With
  one run a task and sampling on, that's a tie.
- **GLM finishes code far sooner.** On the coding tasks it wrote 4 (2.05) to 11 (3.05) times fewer tokens than
  Qwen, which at xhigh thinks for 17-42K tokens on a coding task and can loop. GLM 2.05 did the three coding tasks
  in 12.3 minutes, Qwen in 45.2.
- **GLM 2.05 is 2.3 times as fast as 3.05 per token and thinks about 3 times as long**, so on code it took the same
  time as 3.05, and on everything together half the time (22 against 40 minutes).
- **Qwen reads long prompts 2.7 times as fast** (6,209 against 2,262 tok/s), and its first token came sooner in
  these tasks (1.7-5.8 s against 3.1-9.7 s). Coding agents send large contexts on every turn. The prompt cache spares most of a follow-up turn, but each
  new file or long tool output is read at that speed.
- **2.05 bpw makes careless mistakes.** Both of GLM 2.05's failures are slips in otherwise right work (a list end,
  a rounding step), the kind a 2-bit quant is expected to add. GLM 3.05's were a logic slip and a timeout.

- **Starting GLM 2.05 squeezed this PC's RAM; it no longer does.** With the 40 GiB RAM cache the manager gives it,
  loading took free RAM down to 1.4 GB, from 54 GB with nothing loaded: the page cache from loading sat on top of
  the cache, and curation's benchmark run had to step aside for it. Since the fixes in the
  [CHANGELOG](../../CHANGELOG.md) ("Starting a model leaves the PC its RAM"), switching to it leaves at least 5.8 GB
  free, and 8.8 GB once it has settled. Its curation then passed, with at least 7.1 GB free throughout: answers
  identical to the reference token for token, 19.7 tok/s and 2,277 tok/s with a 38 GiB RAM cache.

One run per task is a small sample: a second run could flip any of these. To settle it, repeat the coding tasks
(about 12 minutes a round on GLM 2.05, 45 on Qwen 4.05 at xhigh), and try Qwen at `medium`, which may keep its
accuracy without the long reasoning.

## Raw results

[results/](results/): each task's answer, reasoning, tokens and times, and the setup it ran on (for the first two
runs, from the manager's log). `python -m repertory results add` logs them for
[docs/benchmarks.md](../../docs/benchmarks.md).

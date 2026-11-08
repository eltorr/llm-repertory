#!/usr/bin/env python3
"""OpenAI-compatible server for GLM-5.3-Flash EXL3 on one 24 GB GPU, over stock exllamav3 1.5.1 (AsyncGenerator).

Offload stack (each piece is a monkeypatch around stock exllamav3, enabled by env, see README):
  exl3_tiers.py    GLM53_ZC_VRAM=0      every routed expert lives in pinned host RAM (home copy, zero-copy readable)
  expert_cache.py  GLM53_EC=1           elastic CLOCK cache of experts in all free VRAM + prefill staging
  cpu_tier.py      GLM53_CPU_TIER=1     coldest decode misses computed by an AVX2 kernel on the host CPU
  triton_pin.py    GLM53_TRITON_PIN     fixed Triton autotune picks for the KDA (linear attention) kernels
  k_hcfuse.py      GLM53_K_HCFUSE=1     fused hyper-connection decode sites (bit-exact)
  k_ftsplit.py     GLM53_K_FTSPLIT=1    faster CPU-tier split kernel (bit-identical)
  k_overlap.py     GLM53_K_OVL=1        shared expert on a side stream in decode (not bitwise: ~2e-3 rel per layer)
  k_lookahead.py   GLM53_LA=1           C1 decode lookahead: next step enqueued on the device token (bit-exact)

Endpoints
  GET  /health, /v1/models, /server_info, /stats
  POST /v1/responses         (OpenAI Responses API, stream or not: reasoning items, function tools -> function_call
                             items, reasoning.effort, previous_response_id; openai_responses.py)
  POST /v1/chat/completions  (Chat Completions on the same pipeline, stream or not; tools -> tool_calls;
                             chat_template_kwargs.enable_thinking / reasoning_effort; reasoning in reasoning_content)
  POST /v1/completions       (raw prompt, stream or not)
  POST /generate, /tokenize  (SGLang-shaped, used by the benchmark and quality tools; return_logprob = teacher-forced;
                             a whole answer comes with its output_ids)
  POST /prepare              (a Chat Completions or Responses body -> {"ids": the prompt's token ids, as the chat
                             routes run it}: for replaying a turn through /generate)
No output caps: max_tokens defaults to the remaining context. A prompt that leaves no room for an answer is refused
before it runs (HTTP 400, code context_length_exceeded), and so is a body that is not a request or a conversation the
chat template refuses (chat_prompt raising ValueError), all in OpenAI's error shape. Usage reports the prompt tokens
the prefix cache served (cached_tokens).

  python3 glm53/serve.py -m /models -cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4 --host 0.0.0.0 --port 30000

As a module (a wrapper server for another model): load the model elsewhere (engines/exl3_offload/loader.py), set the
globals load() sets (args, model, config, cache, tok, draft_model, draft_cache, EOS, hf_tok, CTX, t_load) and replace
what is model-specific: chat_prompt, parse_tool_calls, openai_sp, startup, run_job, AsyncJob, and the tags THINK_OPEN,
THINK_CLOSE, TOOL_OPEN. Everything here looks them up when a request comes in, so assignments take effect.
"""
import argparse, asyncio, contextvars, json, os, re, sys, time, uuid
import torch
from exllamav3 import model_init
from exllamav3.generator import AsyncGenerator, AsyncJob
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offload_stack, stream_split
from openai_compat import normalize_chat_request
from openai_responses import RequestError, ResponseBuilder, responses_to_chat

app = FastAPI()
G = {"gen": None, "stats": {"requests": 0, "completion_tokens": 0}}
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"   # reasoning block (GLM-5.3, Qwen3.8)
TOOL_OPEN = "<tool_call>"                          # start of a tool call: never streamed as content


def load():
    """All model loading lives here (nothing heavy at import time)."""
    global args, model, config, cache, tok, draft_model, draft_cache, EOS, hf_tok, CTX, t_load
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--served-name", default="glm-5.3-flash")
    ap.add_argument("--max-q-size", type=int, default=8)
    args = ap.parse_args()

    offload_stack.pre_load()      # the offload stack's loader patches, by GLM53_* flags (offload_stack.py)
    t_load = time.time()
    model, config, cache, tok, draft_model, _draft_config, draft_cache = model_init.init(args)
    offload_stack.post_load(model)
    t_load = time.time() - t_load
    print(f" -- loaded in {t_load:.1f} s", flush=True)

    EOS = set()
    for e in (getattr(config, "eos_token_id_list", None) or []):
        EOS.add(int(e))
    if tok.eos_token_id is not None:
        EOS.add(int(tok.eos_token_id))
    try:
        gc = json.load(open(os.path.join(args.model_dir, "generation_config.json")))
        for e in (gc.get("eos_token_id") if isinstance(gc.get("eos_token_id"), list) else [gc.get("eos_token_id")]):
            if e is not None:
                EOS.add(int(e))
    except Exception:
        pass
    EOS = sorted(EOS)

    hf_tok = None
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    except Exception as e:
        print(f" !! HF tokenizer unavailable ({e}); chat endpoint uses the built-in GLM template", flush=True)
    CTX = args.cache_size


def normalize_messages(messages):
    """OpenAI -> chat template (GLM-5.3's, and templates like it): assistant tool-call arguments as dicts, content
    never None."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {"arguments": fn["arguments"]}
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def chat_prompt(messages, enable_thinking=True, tools=None, template_kwargs=None):
    if hf_tok is not None and getattr(hf_tok, "chat_template", None):
        kw = dict(template_kwargs or {})
        kw.pop("enable_thinking", None)
        p = hf_tok.apply_chat_template(normalize_messages(messages), tools=tools or None, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=enable_thinking, **kw)
        # GLM-5.3's template ends the generation prompt with "<think>"; thinking off = empty think block
        if not enable_thinking and p.endswith(THINK_OPEN):
            p += THINK_CLOSE
        return p
    p = "[gMASK]<sop>"
    for m in messages:
        p += f"<|{m['role']}|>{m.get('content') or ''}"
    return p + ("<|assistant|><think>" if enable_thinking else "<|assistant|><think></think>")


TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
TOOL_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def parse_tool_calls(content, tools):
    """GLM-5.3 tool calls: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>. String-typed
    parameters (per the tool's JSON schema) stay raw text, everything else is parsed as JSON when it parses."""
    if not tools or "<tool_call>" not in content:
        return content, []
    schema = {}
    for t in tools:
        fn = t.get("function", t)
        schema[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for body in TOOL_CALL.findall(content):
        name = body.split("<arg_key>", 1)[0].strip()
        args = {}
        for k, v in TOOL_ARG.findall(body):
            k, v = k.strip(), v.strip()
            if (schema.get(name, {}).get(k) or {}).get("type") == "string":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except ValueError:
                    args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return TOOL_CALL.sub("", content).strip(), calls


def make_sampler(sp):
    t = float(sp.get("temperature", 1.0) if sp.get("temperature") is not None else 1.0)
    if t <= 0:
        return ArgmaxSampler()
    return ComboSampler(temperature=t, top_p=float(sp.get("top_p") or 1.0), top_k=int(sp.get("top_k") or 0),
                        min_p=float(sp.get("min_p") or 0.0))


@app.on_event("startup")
async def _on_startup():
    r = startup()   # the module global at start time: a wrapper's own startup when it replaced it
    if asyncio.iscoroutine(r):
        await r


async def startup():
    G["gen"] = AsyncGenerator(model=model, cache=cache, tokenizer=tok, max_batch_size=args.max_batch_size,
                              max_chunk_size=args.chunk_size, max_q_size=args.max_q_size,
                              draft_model=draft_model, draft_cache=draft_cache,
                              num_draft_tokens=args.num_draft_tokens, dynamic_draft_tokens=args.dynamic_draft,
                              draft_confidence=args.draft_confidence, ngram_match_min=args.ngram_match_min,
                              record_draft_stats=bool(draft_model is not None or args.ngram_match_min),
                              recurrent_cache_size=int(args.recurrent_cache_size * 1024 ** 3))


def ids_of(body):
    if body.get("input_ids") is not None:
        return torch.tensor([body["input_ids"]], dtype=torch.long)
    return tok.encode(body["text"], encode_special_tokens=True)


class ContextFull(ValueError):
    """The prompt fills the context: no room is left for an answer (HTTP 400)."""
    code = "context_length_exceeded"
    public = True       # written for the client: request_error returns its message

    def __init__(self, prompt_tokens, context):
        super().__init__(f"the prompt has {prompt_tokens:,} tokens and this server's context holds {context:,}, which "
                         f"leaves no room for an answer: shorten the conversation")


def answer_room(prompt_tokens):
    """Tokens an answer may still take: the context after the prompt, the drafts' share (8 when there are none) and 2
    spare."""
    return CTX - prompt_tokens - (args.num_draft_tokens or 8) - 2


def check_room(ids):
    """The room an answer to `ids` has; ContextFull when there is none. Routes call it before they answer (a stream
    can't turn into an error once it has started), and run_job again for any other caller."""
    room = answer_room(ids.shape[-1])
    if room < 1:
        raise ContextFull(ids.shape[-1], CTX)
    return room


CACHED = contextvars.ContextVar("cached_tokens", default=0)   # set by run_job, in the request's own context
OUTPUT = contextvars.ContextVar("output_ids", default=())     # likewise


def cached_tokens():
    """How many of the last job's prompt tokens the prefix cache served (run_job sets it when the answer ends)."""
    return CACHED.get()


def output_ids():
    """The last job's answer as token ids, the token that ended it included (run_job collects them)."""
    return list(OUTPUT.get())


async def run_job(ids, sp, stop_strings=None):
    """Yields (text so far, completion tokens, finish or None) once per generator step. ContextFull before anything
    runs when the prompt leaves no room for an answer; when it ends, cached_tokens() has its prefix-cache hits and
    output_ids() its token ids."""
    gen = G["gen"]
    room = check_room(ids)
    max_new = sp.get("max_new_tokens")
    max_new = room if max_new is None else min(int(max_new), room)
    # N119: GLM53_MAX_RQ_TOKENS = page-allocation round per job (exllamav3 max_rq_tokens: the job requeues itself after
    # that many tokens and keeps going; NOT an output cap). Without it every job reserves pages for the whole remaining
    # context, so a second concurrent job never starts and C2/C4 run one stream after another.
    rq = int(os.environ.get("GLM53_MAX_RQ_TOKENS", "0")) or None
    CACHED.set(0)
    out = []
    OUTPUT.set(out)
    job = AsyncJob(gen, input_ids=ids, max_new_tokens=max(1, max_new), sampler=make_sampler(sp),
                   stop_conditions=list(EOS) + list(stop_strings or []), decode_special_tokens=False,
                   **({"max_rq_tokens": rq} if rq else {}))
    G["stats"]["requests"] += 1
    n, text, done = 0, "", False
    try:
        async for r in job:
            if r.get("stage") != "streaming":
                continue
            chunk = r.get("text") or ""
            tids = r.get("token_ids")
            n += int(tids.numel()) if tids is not None else (1 if chunk else 0)
            if tids is not None:
                out += tids.flatten().tolist()
            text += chunk
            fin = None
            if r.get("eos"):
                # a stop token ends the answer held back (with anything held just before it): exllamav3's "held"
                held = (r.get("held") or {}).get("token_ids")
                if held is not None:
                    out += held.flatten().tolist()
                # exllamav3's new_tokens restarts at every max_rq_tokens requeue (GLM53_MAX_RQ_TOKENS), so for answers longer
                # than one page round it under-reports; the streamed count n covers the whole answer
                n = max(n, int(r.get("new_tokens", 0)))
                # whole KV pages and the matched part of the next, counted from the first round across requeues
                CACHED.set(int(r.get("cached_tokens") or 0))
                reason = r.get("eos_reason")
                fin = {"type": "length" if reason == "max_new_tokens" else "stop", "reason": str(reason)}
                G["stats"]["completion_tokens"] += n
                done = True
            yield text, n, fin
        done = True
    finally:
        if not done:   # client went away: stop the job, free the slot
            try:
                await job.cancel()
            except Exception:
                pass


def teacher_forced(ids, start, topn, want_ids):
    """One forward pass over ids, no cache: for each token from `start` on, the distribution that predicted it (its
    top `topn`, the ids in want_ids, and the token itself), and the one after the last token. Only the rows from
    start - 1 on reach the logits head (last_tokens_only), so the prompt before them costs no logits memory; they are
    sliced here too, for a forward that gives every row."""
    L = ids.shape[-1]
    first = min(max(start - 1, 0), L - 1)
    with torch.inference_mode():
        logits = model.forward(ids, {"attn_mode": "flash_attn_nc", "last_tokens_only": L - first})[0, first - L:]
        logits = logits.float()
    lp = torch.log_softmax(logits, -1)

    def dist(row):
        v, ix = lp[row - first].topk(topn)
        top = [[float(a), int(b), None] for a, b in zip(v.tolist(), ix.tolist())]
        sel = [[float(lp[row - first, t]), int(t), None] for t in want_ids] if want_ids else []
        return top, sel
    in_top, in_ids, in_tok = [], [], []
    for t in range(start, L):
        if t == 0:
            in_top.append(None); in_ids.append(None); in_tok.append(None); continue
        a, b = dist(t - 1); in_top.append(a); in_ids.append(b)
        tid = int(ids[0, t]); in_tok.append([float(lp[t - 1 - first, tid]), tid, None])
    o_top, o_ids = dist(L - 1)
    return {"input_token_logprobs": in_tok, "input_top_logprobs": in_top, "input_token_ids_logprobs": in_ids,
            "output_top_logprobs": [o_top], "output_token_ids_logprobs": [o_ids],
            "next_token": int(lp[L - 1 - first].argmax())}


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


DISC_S = float(os.environ.get("GLM53_DISCONNECT_CHECK_S", "0.1"))


def watch_disconnect(req):
    """Client-disconnect flag polled by a side task. Awaiting req.is_disconnected() inside a streaming loop suspends the
    stream coroutine once per event, and the generator task then runs a whole decode step before the next event goes
    out: streams were delivered at ~half the generation rate with the backlog flushed at the end (G072). The stream
    loops only read the flag; the side task polls every GLM53_DISCONNECT_CHECK_S (0.1 s)."""
    gone = asyncio.Event()

    async def watch():
        while not gone.is_set():
            if await req.is_disconnected():
                gone.set()
                return
            await asyncio.sleep(DISC_S)
    return gone, asyncio.create_task(watch())


# ---- SGLang-shaped endpoints (benchmark / quality tools) ----------------------------------------------------------

@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    sp = body.get("sampling_params") or {}
    ids = ids_of(body)
    if body.get("return_logprob"):
        while len(G["gen"].jobs):
            await asyncio.sleep(0.05)
        r = teacher_forced(ids, int(body.get("logprob_start_len", 0)), int(body.get("top_logprobs_num", 20)),
                           body.get("token_ids_logprob") or [])
        meta = {k: r[k] for k in ("input_token_logprobs", "input_top_logprobs", "input_token_ids_logprobs",
                                  "output_top_logprobs", "output_token_ids_logprobs")}
        meta.update({"prompt_tokens": ids.shape[-1], "completion_tokens": 1, "finish_reason": {"type": "length"}})
        return JSONResponse({"text": "", "output_ids": [r["next_token"]], "meta_info": meta})
    check_room(ids)
    P = ids.shape[-1]
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp)
            gone, wt = watch_disconnect(req)
            try:
                async for text, n, fin in g:
                    yield sse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})
                    if fin is None and gone.is_set():
                        break
                yield "data: [DONE]\n\n"
            finally:
                gone.set(); wt.cancel()
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp):
        pass
    return JSONResponse({"text": text, "output_ids": output_ids(),
                         "meta_info": {"prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}})


@app.post("/tokenize")
async def tokenize(req: Request):
    body = await req.json()
    return JSONResponse({"tokens": tok.encode(body["prompt"], encode_special_tokens=True)[0].tolist()})


# ---- OpenAI-compatible endpoints ----------------------------------------------------------------------------------
# One pipeline behind both APIs: request_body -> prepare_turn (a chat-shaped request as the model runs it) ->
# turn_events (streamed) or turn_answer (whole) -> a formatter. The Responses API (openai_responses.py translates its
# requests into the chat shape and builds its output) is the main one; Chat Completions formats the same turn for the
# many clients that speak it.

def openai_sp(body):
    return {"temperature": body.get("temperature", 1.0), "top_p": body.get("top_p", 1.0), "top_k": body.get("top_k"),
            "min_p": body.get("min_p"), "max_new_tokens": body.get("max_completion_tokens") or body.get("max_tokens")}


def stops_of(body):
    s = body.get("stop")
    return [s] if isinstance(s, str) else list(s or [])


def request_error(e):
    """OpenAI's 400 for a request this server can't run, from the ValueError preparing it raised. Errors written for
    the client (their class says `public`: RequestError for the body, ContextFull, a chat template's refusal) go back
    as they are; any other one gets a generic message, and its text goes to this server's log instead."""
    if getattr(type(e), "public", False):
        message = str(e)
    else:
        print(f"request refused: {type(e).__name__}: {e}", flush=True)
        message = "invalid request"
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "param": None,
                                   "code": getattr(e, "code", None)}}, status_code=400)


@app.exception_handler(ContextFull)
async def _context_full(req, e):
    return request_error(e)   # run_job's own check, for a caller that did not check first


async def request_body(req, needs):
    """The request's JSON object; RequestError (a 400) for anything else, or for an object without `needs`."""
    try:
        body = await req.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or needs not in body:
        raise RequestError(f"expected a JSON object with {needs}")
    return body


def prepare_turn(body):
    """A chat-shaped request (a Responses one after responses_to_chat, or Chat Completions as sent) as the model runs
    it: the prompt's ids, tools, sampling, stops. ValueError for one it can't: chat_prompt may raise one (a template
    that refuses the conversation), and ContextFull when the prompt leaves no room for an answer."""
    body = normalize_chat_request(body)
    kw = dict(body.get("chat_template_kwargs") or {})
    if body.get("reasoning_effort") is not None:
        kw.setdefault("reasoning_effort", body["reasoning_effort"])
    thinking = bool(kw.get("enable_thinking", True))
    tools = body.get("tools") if body.get("tool_choice") != "none" else None
    prompt = chat_prompt(body["messages"], thinking, tools, kw)
    ids = tok.encode(prompt, encode_special_tokens=True)
    check_room(ids)
    return {"body": body, "tools": tools, "ids": ids,
            "in_think": stream_split.prompt_opens_think(prompt, THINK_OPEN, THINK_CLOSE),   # the answer starts as reasoning
            "sp": openai_sp(body), "stops": stops_of(body)}


async def turn_events(req, c):
    """A streamed answer as events: ("reasoning", text) and ("content", text) as they arrive, then ("tool_calls",
    calls) if it called tools, and last ("end", finish_reason, completion_tokens)."""
    g = run_job(c["ids"], c["sp"], c["stops"])
    gone, wt = watch_disconnect(req)
    sent, n, fin, tools = 0, 0, None, c["tools"]
    # reasoning / content / tool calls; a possible partial </think> or <tool_call> is held back until known
    split = stream_split.StreamSplit(c["in_think"], bool(tools), THINK_OPEN, THINK_CLOSE, TOOL_OPEN)
    try:
        async for text, n, fin in g:
            for ev in split.feed(text[sent:]):
                yield ev
            sent = len(text)
            if fin is None and gone.is_set():
                break
        for ev in split.finish():
            yield ev
        reason = (fin or {}).get("type", "stop")
        if tools and split.tool_text:   # everything from the first tool-call tag on, parsed once at the end
            content, calls = parse_tool_calls(split.tool_text, tools)
            if content:
                yield ("content", content)
            if calls:
                reason = "tool_calls"
                yield ("tool_calls", calls)
        yield ("end", reason, n)
    finally:
        gone.set(); wt.cancel()
        await g.aclose()


async def turn_answer(c):
    """The whole answer: (reasoning, content, tool calls, finish_reason, completion_tokens)."""
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(c["ids"], c["sp"], c["stops"]):
        pass
    # a think block that never closed (e.g. hit max_tokens) is all reasoning
    reasoning, content = stream_split.split_answer(text, c["in_think"], THINK_OPEN, THINK_CLOSE)
    content, calls = parse_tool_calls(content, c["tools"])
    return reasoning, content, calls, "tool_calls" if calls else (fin or {}).get("type"), n


@app.post("/v1/responses")
async def responses(req: Request):
    """OpenAI's Responses API: the request as a chat-shaped turn, the answer as Responses output
    (openai_responses.py)."""
    try:
        raw = await request_body(req, "input")
        chat_body, convo = responses_to_chat(raw)
        c = prepare_turn(chat_body)
    except ValueError as e:   # RequestError, a template's refusal, ContextFull
        return request_error(e)
    P = c["ids"].shape[-1]
    rb = ResponseBuilder(raw, args.served_name, convo)
    if raw.get("stream"):
        async def events():
            for e in rb.start():
                yield e
            evs = turn_events(req, c)
            try:
                async for ev in evs:
                    for e in rb.feed(ev, P, cached_tokens()):
                        yield e
            finally:
                await evs.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    reasoning, content, calls, reason, n = await turn_answer(c)
    return JSONResponse(rb.whole(reasoning, content, calls, reason, P, n, cached_tokens()))


def chat_usage(P, n):
    return {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n,
            "prompt_tokens_details": {"cached_tokens": cached_tokens()}}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    """Chat Completions: the same turn as chat.completion objects, or chat.completion.chunk events."""
    try:
        c = prepare_turn(await request_body(req, "messages"))
    except ValueError as e:   # RequestError, a template's refusal, ContextFull
        return request_error(e)
    cid, created, P = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), c["ids"].shape[-1]
    if c["body"].get("stream"):
        async def events():
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.served_name}
            delta = lambda d: sse(dict(base, choices=[{"index": 0, "delta": d, "finish_reason": None}]))
            key = {"reasoning": "reasoning_content", "content": "content"}
            yield delta({"role": "assistant", "content": ""})
            evs = turn_events(req, c)
            try:
                async for ev in evs:
                    if ev[0] == "end":
                        yield sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ev[1]}],
                                       usage=chat_usage(P, ev[2])))
                        yield "data: [DONE]\n\n"
                    elif ev[0] == "tool_calls":
                        yield delta({"tool_calls": [dict(t, index=i) for i, t in enumerate(ev[1])]})
                    else:
                        yield delta({key[ev[0]]: ev[1]})
            finally:
                await evs.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    reasoning, content, calls, reason, n = await turn_answer(c)
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        msg["tool_calls"] = calls
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": args.served_name,
                         "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
                         "usage": chat_usage(P, n)})


@app.post("/prepare")
async def prepare(req: Request):
    """The prompt's token ids for a Chat Completions body (messages) or a Responses one (input), from the chat routes'
    own prepare_turn: the same template, tools and thinking setting. For replaying a turn through /generate as
    token ids; like /generate, not an OpenAI route."""
    try:
        try:
            body = await req.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and "messages" not in body and "input" in body:
            body = responses_to_chat(body)[0]
        if not isinstance(body, dict) or "messages" not in body:
            raise RequestError("expected a JSON object with messages or input")
        c = prepare_turn(body)
    except ValueError as e:   # RequestError, a template's refusal, ContextFull
        return request_error(e)
    return JSONResponse({"ids": c["ids"][0].tolist()})


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = tok.encode(prompt, encode_special_tokens=True)
    check_room(ids)
    sp, stops = openai_sp(body), stops_of(body)
    cid, created, P = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), ids.shape[-1]
    base = {"id": cid, "object": "text_completion", "created": created, "model": args.served_name}
    if body.get("stream"):
        async def events():
            g = run_job(ids, sp, stops)
            gone, wt = watch_disconnect(req)
            sent, n, fin = 0, 0, None
            try:
                async for text, n, fin in g:
                    if len(text) > sent:
                        yield sse(dict(base, choices=[{"index": 0, "text": text[sent:], "finish_reason": None}]))
                        sent = len(text)
                    if fin is None and gone.is_set():
                        break
                yield sse(dict(base, choices=[{"index": 0, "text": "", "finish_reason": (fin or {}).get("type", "stop")}],
                               usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}))
                yield "data: [DONE]\n\n"
            finally:
                gone.set(); wt.cancel()
                await g.aclose()
        return StreamingResponse(events(), media_type="text/event-stream")
    text, n, fin = "", 0, None
    async for text, n, fin in run_job(ids, sp, stops):
        pass
    return JSONResponse(dict(base, choices=[{"index": 0, "text": text, "finish_reason": (fin or {}).get("type")}],
                             usage={"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n}))


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local",
                                                     "max_model_len": CTX}]})


@app.get("/server_info")
async def server_info():
    sa = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
    sa.update({"context_length": CTX, "max_running_requests": args.max_batch_size, "max_total_num_tokens": CTX,
               "chunked_prefill_size": args.chunk_size, "kv_cache_dtype": args.cache_quant or "fp16",
               "engine": "exllamav3", "env": {k: v for k, v in os.environ.items() if k.startswith(("GLM53_", "EXL3_"))}})
    return JSONResponse({"context_length": CTX, "server_args": sa, "load_seconds": round(t_load, 1)})


@app.get("/stats")
async def stats():
    st = dict(G["stats"])
    if os.environ.get("GLM53_EC"):
        import expert_cache
        st["expert_cache"] = expert_cache.summary()
    if os.environ.get("GLM53_CPU_TIER") == "1":
        import cpu_tier
        st["cpu_tier"] = cpu_tier.summary()
    if os.environ.get("GLM53_K_OVL") == "1":
        import k_overlap
        st["k_overlap"] = k_overlap.summary()
    if os.environ.get("GLM53_K_HCFUSE") == "1":
        import k_hcfuse
        st["k_hcfuse"] = dict(k_hcfuse.STATS)
    if os.environ.get("GLM53_LA") == "1":
        import k_lookahead
        st["k_lookahead"] = k_lookahead.summary()
    if os.environ.get("GLM53_NV") == "1":
        import nv_tier
        st["nv_tier"] = nv_tier.summary()
    if os.environ.get("GLM53_NV") == "2":
        import nv2
        st["nv2"] = nv2.summary()
    return JSONResponse(st)


@app.get("/nv_trace")
async def nv_trace():
    import nv_tier
    return JSONResponse(nv_tier.save_trace())


@app.get("/nv_verify")
async def nv_verify(n: int = 64):
    if os.environ.get("GLM53_NV") == "2":
        import nv2
        return JSONResponse(nv2.NV.verify(n))
    import nv_tier
    return JSONResponse(nv_tier.NV.verify(n) if nv_tier.NV is not None else {"error": "GLM53_NV off"})


@app.get("/la")
async def la(on: int = 1):
    """N136: runtime lookahead switch for same-server A/B (takes effect at the next decode step)."""
    import k_lookahead
    return JSONResponse({"enabled": k_lookahead.set_enabled(bool(on)), "stats": k_lookahead.summary()})


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    load()
    if os.environ.get("GLM53_MAIN_CPUS"):   # N119: GPU-feeding thread off the CPU-tier cores
        import nv2
        os.sched_setaffinity(0, set(nv2._cpus(os.environ["GLM53_MAIN_CPUS"])))
        torch.set_num_threads(int(os.environ.get("GLM53_TORCH_THREADS", "1")))   # no spinning OpenMP pool on the tier cores
        print(f" -- main thread on cpus {sorted(os.sched_getaffinity(0))}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

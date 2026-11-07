"""The exl3_offload engine's model server, as Repertory runs it.

    python engines/exl3_offload/server.py --model ./model --store ./store/qwen.bin --ram-gb 34 [--cache 262144]
                                          [--chunk 24576] [--host 127.0.0.1] [--port 30000] [--mtp N] [--cpu-lane]

engines/exl3_offload/run.sh starts it sized for the card; the manager starts it through bin/repertory-backend.

The model is loaded by loader.py and served by the engine's own app (glm53-flash-offload/glm53/serve.py): its routes
answer every request, /v1/responses and /v1/chat/completions included, with the context check, the prefix-cache
figures and OpenAI's 400s. This file adds what is Repertory's or the model's:
  - the model's format in the engine's hooks (qwen_chat.py): its chat template, its tool calls, its sampling values
    where a request leaves them out, and its end-of-turn token among the stop tokens;
  - /memory: the RAM and VRAM watchdogs (memory_guard.py);
  - the manager's switches without a restart: POST /expert_cache/resize, GET /expert_cache/ram and
    POST /expert_cache/ram/stop;
  - LocalOnly: no web pages, and only the manager when it set a key.
"""
import argparse
import asyncio
import hmac
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "glm53-flash-offload" / "glm53"
LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")
KEY_HEADER = b"x-repertory-key"

sys.path.insert(0, str(HERE))
sys.path.append(str(ENGINE))            # the engine's modules: serve, nv2, expert_cache...
import memory_guard  # noqa: E402
import qwen_chat  # noqa: E402


# ------------------------------------------------------------------------------------------------------- live resize
RESIZE_KEYS=('vram_limit_bytes','max_slots','reserve_bytes','elastic_bytes','torch_limit_bytes','chunk','ram_bytes')
GIB=1024**3

def resize_args(body):
    """Checks a /expert_cache/resize body. Every field is optional and a whole non-negative number; the limits and the
    RAM cache size must be positive, and chunk a multiple of the 256-token page."""
    if not isinstance(body,dict):raise ValueError('expected a JSON object')
    extra=sorted(set(body)-set(RESIZE_KEYS))
    if extra:raise ValueError('unknown fields: '+', '.join(extra))
    out={}
    for k in RESIZE_KEYS:
        v=body.get(k)
        if v is None:continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<2**50 or v!=int(v):
            raise ValueError(f'{k} must be a whole number of at least 0')
        out[k]=int(v)
    if 0 in (out.get('vram_limit_bytes'),out.get('torch_limit_bytes'),out.get('elastic_bytes'),out.get('ram_bytes')):
        raise ValueError('the limits, elastic_bytes and ram_bytes must be positive')
    c=out.get('chunk')
    if c is not None and not (256<=c<=65536 and c%256==0):
        raise ValueError('chunk must be a multiple of 256, up to 65536')
    return out

def move_ram_limits(host,delta):
    """The RAM cache changes by delta bytes: the memory cgroup's max and high (run.sh: the RAM cache plus a margin)
    and the RAM guard's limit move with it, raised before it grows and lowered after it shrinks, never below what is
    in use plus 2 GiB. False if a limit could not be raised (the cache must not grow then; nothing is changed); one
    that can't be lowered just stays higher."""
    if not delta:return True
    used=host.sample();done=[]
    files=[host.path/'memory.max',host.path/'memory.high']
    for f in files if delta>0 else files[::-1]:
        try:old=f.read_text().strip()
        except OSError:continue
        if old=='max':continue
        new=int(old)+delta if delta>0 else max(int(old)+delta,used+2*GIB)
        try:f.write_text(str(new));done.append((f,old))
        except OSError:
            if delta<0:continue
            for g,v in reversed(done):
                try:g.write_text(v)
                except OSError:pass
            return False
    if host.limit is not None:
        host.limit=host.limit+delta if delta>0 else max(host.limit+delta,used+2*GIB)
    return True

def compact_memory():
    """After the RAM cache shrinks under WSL: free memory goes back to Windows in whole blocks (free page reporting),
    so compaction hands it back sooner. QWEN_RAM_COMPACT=0 turns it off; it needs root."""
    try:
        with open('/proc/sys/vm/compact_memory','w') as f:f.write('1')
    except OSError:pass

def in_wsl():
    try:return 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:return False

def mem_available():
    """This system's available RAM in bytes (inside WSL, the VM's), or None."""
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    except (OSError,ValueError,IndexError):pass
    return None

class RamWorker:
    """The RAM cache's slow halves, off the request path and one job at a time: growing it a chunk at a time, and
    handing the chunks a shrink let go of back to the system. The memory limits follow what it really adds or frees.
    A grow stops early when this system runs short of RAM (QWEN_RAM_GROW_FLOOR_GB left, 4), when told to (wait with
    cancel), or when a new RAM resize comes.
    Requests come first: the CUDA driver serializes pinning and unpinning memory with the model's GPU work (under WSL
    pinning a 0.3 GB chunk takes ~0.13 s, and a request waited out a whole 27 GiB grow). So while one runs (busy), a
    grow adds a small step (BUSY_STEP slots) once every BUSY_GROW_S, and a free hands back a chunk once every
    BUSY_FREE_S (sooner: the memory was freed because something else needs it); between requests both go at full
    speed.
    Memory is reused: the chunks a shrink takes out stay pending until they are freed, the highest first, and a grow
    that comes meanwhile (pressure that let up again) takes the pending ones back before it pins anything: they are
    still pinned, and the limits never came down for them. nv: the nv2 engine (nram, slot, ram.cs, grow_ram_step,
    free_ram_chunks, reattach_ram_chunks); available: what mem_available reads; busy: whether a request is running
    or waiting."""
    BUSY_GROW_S,BUSY_STEP,BUSY_FREE_S=1.0,32,0.1
    def __init__(self,nv,host,available=mem_available,busy=lambda:False):
        self.nv,self.host,self.available,self.busy=nv,host,available,busy
        self.floor=int(float(os.environ.get('QWEN_RAM_GROW_FLOOR_GB','4'))*GIB)
        self.thread=None;self.cancel=threading.Event();self.lock=threading.Lock()
        self.pending=[]                         # (buffer, address, slots) out of the cache, not freed yet: in order
        self.state={'growing':False,'freeing':False,'target_slots':nv.nram,'stopped':None,'reused_slots':0}
    def wait(self,cancel=False):
        """Until the job in progress ends; cancel: a grow stops after the chunk it is pinning, a free after the one it
        is handing back (the rest stay pending: reuse takes them, or free_pending)."""
        if cancel:self.cancel.set()
        if self.thread is not None:self.thread.join()
        self.thread=None;self.cancel.clear()
    def info(self):
        nv=self.nv
        with self.lock:pend=sum(c[2] for c in self.pending)
        return dict(self.state,ram_slots=nv.nram,ram_bytes=nv.nram*nv.slot,pending_bytes=pend*nv.slot,
                    target_bytes=self.state['target_slots']*nv.slot,limit_bytes=self.host.limit)
    def _start(self,fn,*a):
        self.thread=threading.Thread(target=fn,args=a,daemon=True,name='ram-cache');self.thread.start()
    def prepare(self,target_slots):
        """A RAM resize to target_slots, with no job running (after wait with cancel) and before the VRAM cache
        changes. Growing: the pending chunks go back in first, up to the target (the last may pass it by part of a
        chunk), and the limits are raised for the rest. False if they could not be raised (nothing grows then)."""
        self.state['reused_slots']=0
        if target_slots<=self.nv.nram:return True
        with self.lock:
            take=[];n=self.nv.nram
            while self.pending and n<target_slots:
                c=self.pending.pop(0);take.append(c);n+=c[2]
        if take:self.state['reused_slots']=self.nv.reattach_ram_chunks(take)
        return self.prepare_grow(target_slots)
    def settle(self,target_slots,shrunk=()):
        """After it: the chunks a shrink took out join the pending ones; then a grow toward target_slots, or the
        pending chunks nothing took back go to the system."""
        if shrunk:
            with self.lock:self.pending[:0]=list(shrunk)         # they sit below the ones already pending
        if not shrunk and target_slots>self.nv.nram:self.grow(target_slots)
        else:
            self.state['target_slots']=target_slots
            self.free_pending()
    def prepare_grow(self,target_slots):
        """Raise the limits by the whole growth, before anything else changes; False if they could not be raised."""
        return move_ram_limits(self.host,max(0,target_slots-self.nv.nram)*self.nv.slot)
    def grow(self,target_slots):
        planned=max(0,target_slots-self.nv.nram)*self.nv.slot
        self.state.update(growing=planned>0,target_slots=target_slots,stopped=None)
        if planned:self._start(self._grow,target_slots,planned)
    def _gap(self,since,every):
        """Wait while requests run, until a gap or until every seconds after since; False if cancelled meanwhile."""
        while self.busy() and time.monotonic()-since<every:
            if self.cancel.wait(0.01):return False
        return not self.cancel.is_set()
    def _grow(self,target,planned):
        nv=self.nv;before=nv.nram;t0=time.monotonic();last=0.0;busy_steps=0
        try:
            while nv.nram<target and self._gap(last,self.BUSY_GROW_S):
                step=None
                if self.busy():step=self.BUSY_STEP;busy_steps+=1
                free=self.available()
                if free is not None and free<self.floor+nv.ram.cs*nv.slot:
                    self.state['stopped']='this system is short of RAM';break
                try:n=nv.grow_ram_step(target,step)
                except Exception as e:  # noqa: BLE001 - a grow that fails keeps what it has
                    self.state['stopped']=f'error: {e}';break
                if not n:
                    self.state['stopped']='pinned memory ran out';break
                last=time.monotonic()
                time.sleep(0.002)               # the request path gets the interpreter between chunks
            if nv.nram<target and not self.state['stopped']:self.state['stopped']='stopped'
        finally:
            move_ram_limits(self.host,(nv.nram-before)*nv.slot-planned)   # lower them by what it fell short of
            self.state.update(growing=False,grow_s=round(time.monotonic()-t0,2),grow_busy_steps=busy_steps)
            print('QWEN_RAM_GROW',json.dumps(self.info()),flush=True)
    def free_pending(self):
        """Hand the pending chunks back to the system in the background, the highest first; the limits come down by
        what was freed (only then: the memory is in use until it is) and, under WSL, free memory is compacted so
        Windows gets it sooner."""
        with self.lock:
            if not self.pending:return
        self.state['freeing']=True
        self._start(self._free)
    def _free(self):
        t0=time.monotonic();last=0.0;freed=0
        try:
            while not self.cancel.is_set():
                while self.busy() and time.monotonic()-last<self.BUSY_FREE_S:
                    if self.cancel.wait(0.01):break
                if self.cancel.is_set():break
                with self.lock:
                    if not self.pending:break
                    c=self.pending.pop()
                try:self.nv.free_ram_chunks([c])
                except Exception:
                    with self.lock:self.pending.append(c)   # still pinned: it stays pending
                    raise
                freed+=c[2]*self.nv.slot;last=time.monotonic()
        finally:
            move_ram_limits(self.host,-freed)
            with self.lock:done=not self.pending
            self.state.update(freeing=False,free_s=round(time.monotonic()-t0,2))
            if freed and done and os.environ.get('QWEN_RAM_COMPACT','1' if in_wsl() else '0')=='1':compact_memory()


# ---------------------------------------------------------------------------------------------------------- the model
def stop_tokens(config, tok, model_dir):
    """Token ids that end an answer: the model config's, the tokenizer's, generation_config.json's and the chat
    format's end of turn."""
    ids = set(getattr(config, "eos_token_id_list", None) or [])
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    try:
        listed = json.loads((Path(model_dir) / "generation_config.json").read_text()).get("eos_token_id")
    except (OSError, ValueError):
        listed = None
    ids.update(listed if isinstance(listed, list) else [] if listed is None else [listed])
    try:
        ids.add(tok.single_id("<|im_end|>"))
    except (KeyError, ValueError, AttributeError):
        pass
    return sorted(int(i) for i in ids)


# ------------------------------------------------------------------------------------------------------------ server
class Server:
    """The engine's app with Repertory's additions, around a loaded model."""

    def __init__(self, cli, engine, loaded, template, ram, vram):
        model, config, cache, tok, self.pools, self.tables, draft_model, draft_cache = loaded
        self.cli, self.engine, self.template, self.ram, self.vram = cli, engine, template, ram, vram
        self.resizing, self.worker = asyncio.Lock(), None
        # what the engine's routes read from the command line serve.load would have parsed
        settings = {"cache_size": cli.cache, "chunk_size": cli.chunk, "cache_quant": "4",
                    "recurrent_cache_size": 0.25,       # GiB of recurrent-state checkpoints: the prefix cache
                    "num_draft_tokens": cli.mtp, "dynamic_draft": False, "draft_confidence": 0.0,
                    "ngram_match_min": 0,
                    "max_batch_size": 1, "max_q_size": 8,   # one answer at a time; the manager queues up to 8 more
                    "model_dir": cli.model, "served_name": cli.served_name}
        engine.args = argparse.Namespace(**settings, host=cli.host, port=cli.port)
        engine.model, engine.config, engine.cache, engine.tok = model, config, cache, tok
        engine.draft_model, engine.draft_cache = draft_model, draft_cache
        engine.EOS, engine.CTX, engine.hf_tok = stop_tokens(config, tok, cli.model), cli.cache, None
        self.wire()

    # -------------------------------------------------------------------------------------- engine hooks
    def wire(self):
        """The model's format in the engine's hooks, wherever it builds a prompt or reads an answer: the chat template
        (its refusals are TemplateError, a ValueError: the engine answers 400), the tool calls and the sampling
        values. Qwen3.8 writes the engine's own think and tool-call tags, so those stay. Then Repertory's routes."""
        e, server = self.engine, self
        plain_sampling = e.openai_sp
        e.openai_sp = lambda body: plain_sampling(qwen_chat.with_sampling(body))
        e.chat_prompt = lambda messages, enable_thinking=True, tools=None, template_kwargs=None: self.template.render(
            e.normalize_messages(messages), tools, enable_thinking, template_kwargs)
        e.parse_tool_calls = qwen_chat.parse_tool_calls
        from fastapi import Request

        async def resize(req: Request):
            return await server.resize(req)

        app = e.app
        app.add_api_route("/memory", self.memory, methods=["GET"])
        app.add_api_route("/expert_cache/ram", self.ram_state, methods=["GET"])
        app.add_api_route("/expert_cache/ram/stop", self.ram_stop, methods=["POST"])
        app.add_api_route("/expert_cache/resize", resize, methods=["POST"])
        app.on_event("shutdown")(self.shutdown)

    async def shutdown(self):
        gen = self.engine.G.get("gen")
        if gen is not None:
            await gen.close()
        for guard in (self.vram, self.ram):
            if guard is not None:
                guard.close()

    # -------------------------------------------------------------------------------------- memory
    async def memory(self):
        def figures(guard):
            if guard is None:
                return None
            return {"used": guard.sample(), "peak": guard.peak, "limit": guard.limit}
        return {"ram": dict(figures(self.ram), cgroup=str(self.ram.path)), "vram": figures(self.vram)}

    # -------------------------------------------------------------------------------------- live resize
    def ram_worker(self):
        import nv2
        gen_jobs = self.engine.G

        def busy():                             # a request running or waiting
            gen = gen_jobs.get("gen")
            return gen is not None and len(gen.jobs) > 0
        if self.worker is None:
            self.worker = RamWorker(nv2.NV, self.ram, busy=busy)
        return self.worker

    async def ram_state(self):
        import nv2
        from fastapi.responses import JSONResponse
        if nv2.NV is None:
            return JSONResponse({'error': 'no resizable RAM cache (nv2 is off)'}, 404)
        return self.ram_worker().info()

    async def ram_stop(self):
        # the brake while it grows and other apps need the RAM: keeps what it has, and waits for no request
        import nv2
        from fastapi.responses import JSONResponse
        if nv2.NV is None:
            return JSONResponse({'error': 'no resizable RAM cache (nv2 is off)'}, 404)
        async with self.resizing:
            w = self.ram_worker()
            await asyncio.to_thread(w.wait, True)
            w.free_pending()                    # a free it stopped goes on
            return w.info()

    async def resize(self, req):
        # The manager's switch between profiles without a restart: the VRAM expert cache, the torch cap and the
        # prefill chunk change between requests, with the model loaded (expert_cache.Pool.resize).
        from fastapi.responses import JSONResponse
        try:
            a = resize_args(json.loads(await req.body() or b'{}'))
        except ValueError as e:
            return JSONResponse({'error': str(e)}, 400)
        import expert_cache
        import nv2
        import torch
        if 'ram_bytes' in a:
            # 'ram_bytes:' errors tell the manager this engine can't resize its RAM cache (it restarts instead)
            if nv2.NV is None:
                return JSONResponse({'error': 'ram_bytes: no resizable RAM cache (nv2 is off)'}, 400)
            if not nv2.NV.can_resize_ram():
                return JSONResponse({'error': 'ram_bytes: the compiled kernels predate RAM resizing; compile them '
                                              'again (repertory setup build)'}, 400)
        async with self.resizing:
            if 'ram_bytes' in a:
                # a grow in progress stops after the chunk it is pinning, a free after the one it is handing back
                # (the rest stay pending, for a grow to take back): off the event loop, so requests keep running
                # meanwhile (only this route, under the lock, starts another job)
                await asyncio.to_thread(self.ram_worker().wait, True)
            gen = self.engine.G['gen']
            deadline = time.monotonic() + 60
            while len(gen.jobs):
                if time.monotonic() > deadline:
                    if 'ram_bytes' in a:
                        self.ram_worker().free_pending()
                    return JSONResponse({'error': 'a request is still running'}, 409)
                await asyncio.sleep(0.05)
            # nothing from here to the response awaits, so no request starts in between
            t0, before = time.monotonic(), memory_guard.device_used()
            target = None
            if 'ram_bytes' in a:
                w, nv = self.ram_worker(), nv2.NV
                target, ram_before = nv.ram_target_slots(a['ram_bytes']), nv.nram
                # growing, it first takes back what a shrink has not freed yet (still pinned: nothing to wait for)
                if not w.prepare(target):
                    w.free_pending()
                    return JSONResponse({'error': 'ram_bytes: the memory cgroup limit could not be raised'}, 400)
            if 'torch_limit_bytes' in a:
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, a['torch_limit_bytes'] / torch.cuda.mem_get_info()[1]))
            pools = expert_cache.resize(limit_bytes=a.get('vram_limit_bytes'), max_slots=a.get('max_slots'),
                                        reserve_bytes=a.get('reserve_bytes'), elastic_bytes=a.get('elastic_bytes'),
                                        measure_used=memory_guard.device_used)
            ram = None
            if target is not None:
                if target < ram_before:
                    # after the VRAM cache: experts it gave up were written back into RAM, so the newest are kept.
                    # Requests wait while the kept experts move; the chunks go back to the system in the background
                    ram, chunks = nv.shrink_ram(a['ram_bytes'])
                    w.settle(target, chunks)
                else:
                    # the rest in empty chunks, pinned in the background while requests run (RamWorker)
                    ram = {'ram_slots_before': ram_before, 'ram_bytes_before': ram_before * nv.slot, 'dropped': 0,
                           'moved': 0, 'ms': 0.0}
                    w.settle(target)
                ram.update(w.info())
            g = gen.generator
            if 'chunk' in a:
                g.max_chunk_size = self.engine.args.chunk_size = a['chunk']
                # checkpoints are taken on chunk boundaries: keep the prompt-prefill interval a multiple of the chunk
                pp = g.recurrent_checkpoint_interval_pp
                if pp % a['chunk']:
                    g.recurrent_checkpoint_interval_pp = -(-pp // a['chunk']) * a['chunk']
            out = {'pools': pools, 'chunk': g.max_chunk_size, 'vram_used_before': before,
                   'vram_used': memory_guard.device_used(), 'ram': ram, 'seconds': round(time.monotonic() - t0, 2)}
            print('QWEN_RESIZE', json.dumps(out), flush=True)
            return out


class LocalOnly:
    """Refuses web pages (any Origin header) and, when the Repertory manager started this server with a key, requests
    without it. Bound to localhost (the default), it also refuses requests addressed to another name (DNS rebinding);
    bound to a network address (--host), it answers any name, as the operator chose."""
    def __init__(self,app,key,local=True):self.app,self.key,self.local=app,key.encode(),local
    async def __call__(self,scope,receive,send):
        if scope['type']=='http':
            h=dict(scope['headers'])
            host=h.get(b'host',b'').decode('latin-1').lower()
            host=host[1:host.find(']')] if host.startswith('[') else host.rsplit(':',1)[0]
            if (b'origin' in h or (self.local and host not in LOCAL_NAMES)
                    or (self.key and not hmac.compare_digest(h.get(KEY_HEADER,b''),self.key))):
                body=b'{"error":"refused: a web page, another host name, or a call without the manager key"}'
                await send({'type':'http.response.start','status':403,'headers':[(b'content-type',b'application/json'),
                            (b'content-length',str(len(body)).encode())]})
                await send({'type':'http.response.body','body':body})
                return
        await self.app(scope,receive,send)


# -------------------------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="The exl3_offload engine's model server (OpenAI-compatible).")
    p.add_argument("--model", required=True, help="the model folder (config, tokenizer, chat template, weights)")
    p.add_argument("--store", required=True, help="the packed expert store (NAME.bin, with NAME.json beside it)")
    p.add_argument("--ram-gb", type=float, required=True, help="GiB of RAM for the expert RAM cache")
    p.add_argument("--cache", type=int, default=262144, help="context window in tokens (default 262144)")
    # smaller cards need smaller prefill chunks: sizing.py picks one, run.sh passes it as QWEN_CHUNK
    p.add_argument("--chunk", type=int, default=int(os.environ.get("QWEN_CHUNK", "24576")),
                   help="prefill chunk in tokens (default QWEN_CHUNK, else 24576)")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this machine only)")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--served-name", default="qwen3.8-flash-next", help="the model name /v1/models lists")
    # Off by default; the environment variables let run.sh and the manager set them.
    p.add_argument("--mtp", type=int, default=int(os.environ.get("QWEN_MTP", "0")),
                   help="Tokens the MTP layer drafts per step (0: off)")
    p.add_argument("--cpu-lane", action="store_true", default=os.environ.get("QWEN_CPU_LANE") == "1",
                   help="Compute RAM-resident experts on the CPU in decode (3-bit experts only)")
    p.add_argument("--prefetch", action="store_true", default=os.environ.get("GLM53_NV_PREFETCH") == "1",
                   help="Start NVMe reads for the next layer's predicted experts early")
    return p.parse_args(argv)


def measure_vram():
    """Before the model loads: GLM53_VRAM_UNSEEN_BYTES, the VRAM other processes hold that CUDA doesn't show this one
    (the expert cache counts it when the manager resizes it to a device-wide limit, as NVML counts VRAM), kernels
    built for this GPU, and the torch cap for the loader (which also bounds the expert cache it starts with)."""
    import torch
    torch.cuda.init()
    device = torch.cuda.current_device()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "%d.%d" % torch.cuda.get_device_capability(device))
    free, total = torch.cuda.mem_get_info(device)
    if os.environ.get("QWEN_VRAM_LIMIT_BYTES"):   # the manager's device-wide limit: the expert cache sizes itself to it
        os.environ.setdefault("GLM53_VRAM_LIMIT_BYTES", os.environ["QWEN_VRAM_LIMIT_BYTES"])
    if "GLM53_VRAM_UNSEEN_BYTES" not in os.environ:
        unseen = memory_guard.unseen_vram(free, total, device)
        if unseen is not None:
            os.environ["GLM53_VRAM_UNSEEN_BYTES"] = str(unseen)
            if unseen > 64 * 1024 ** 2:
                print(f"VRAM: other processes use {unseen / GIB:.2f} GiB that CUDA does not report here; counted "
                      f"against the VRAM limit", flush=True)
    return memory_guard.torch_limit_bytes(total)


def main(argv=None):
    cli = parse_args(argv)
    if cli.prefetch:
        os.environ["GLM53_NV_PREFETCH"] = "1"
    try:
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)    # kill -USR1 PID prints every thread's stack
    except (AttributeError, ImportError, ValueError):
        pass
    vram = memory_guard.VramGuard()
    ram = memory_guard.RamGuard(os.environ.get("QWEN_GLOBAL_MEMORY_CGROUP",
                                               os.environ.get("QWEN_MEMORY_CGROUP", "/sys/fs/cgroup")))
    torch_limit = measure_vram()
    import loader
    t0 = time.monotonic()
    loaded = loader.load(cli.model, cli.store, chunk=cli.chunk, cache_tokens=cli.cache, ram_gb=cli.ram_gb,
                         cpu=cli.cpu_lane, mtp=cli.mtp, embed_disk=True, cache_quant=4, torch_limit_bytes=torch_limit)
    sys.path.insert(0, str(ENGINE))
    import serve as engine
    engine.t_load = time.monotonic() - t0
    Server(cli, engine, loaded, qwen_chat.ChatTemplate.from_model(cli.model), ram, vram)
    print(f"ENGINE READY in {engine.t_load:.0f} s: context {cli.cache:,} tokens, {cli.ram_gb:g} GiB RAM cache",
          flush=True)
    import uvicorn
    # no access log: request paths are client text, and the manager reads this log for out-of-memory messages
    uvicorn.run(LocalOnly(engine.app, os.environ.get("REPERTORY_BACKEND_KEY", ""), cli.host in LOCAL_NAMES),
                host=cli.host, port=cli.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()

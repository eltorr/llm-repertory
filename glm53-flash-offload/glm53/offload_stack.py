"""The offload stack's hooks around a stock exllamav3 model load, in the order serve.py has always run them, for
serve.py and for other loaders (engines/exl3_offload/loader.py). Each piece is enabled by its GLM53_* flag (README,
docs/how-it-works.md); nothing here reads command-line arguments.

  pre_load()        before exllamav3 builds and loads the model: loader patches (NVMe tier, BC capture of the expert
                    cache, shared expert out of the fused kernel, Triton pins)
  post_load(model)  after the model (and its draft model) are loaded: CPU tier wrap, expert cache + NVMe tier attach,
                    kernels; returns the expert cache's pools (expert_cache.POOLS, [] when GLM53_EC is off)
"""
import os


def pre_load():
    if os.environ.get("GLM53_ZC_VRAM"):
        import exl3_tiers
        exl3_tiers.install()
    if os.environ.get("GLM53_NV") in ("1", "2"):   # N116/N119 NVMe tier: loader skips the routed-expert bytes
        assert not os.environ.get("GLM53_ZC_VRAM"), "GLM53_NV replaces the pinned home copy (unset GLM53_ZC_VRAM)"
        import nv_tier
        nv_tier.install()
    if os.environ.get("GLM53_EC"):
        import expert_cache
        expert_cache.install()
    if os.environ.get("GLM53_K_OVL") == "1":   # shared experts out of the fused decode kernel (k_overlap.py)
        import k_overlap
        k_overlap.install()
    if os.environ.get("GLM53_TRITON_PIN"):
        import triton_pin
        triton_pin.install()


def post_load(model):
    pools = []
    if os.environ.get("GLM53_LA") == "1":   # N136: embedding table host-mapped (device-side lookup for lookahead)
        import k_lookahead
        k_lookahead.prep_model(model)
    if os.environ.get("GLM53_ZC_VRAM"):
        import exl3_tiers
        print(f" -- zero-copy tier summary: {exl3_tiers.summary()}", flush=True)
    if os.environ.get("GLM53_CPU_TIER") == "1":   # routers wrapped BEFORE the cache attaches
        import cpu_tier
        cpu_tier.wrap(model)
    if os.environ.get("GLM53_EC"):
        import expert_cache
        pools = expert_cache.attach(model)
    if os.environ.get("GLM53_NV") == "1":
        import nv_tier
        nv_tier.finish(model)
    if os.environ.get("GLM53_NV") == "2":
        import nv2
        nv2.finish(model)
    if os.environ.get("GLM53_CPU_TIER") == "1":   # register the pinned home copies, self-test, start the worker
        import cpu_tier
        cpu_tier.start()
        if os.environ.get("GLM53_K_FTSPLIT") == "1":   # faster split kernel, bit-identical outputs (k_ftsplit.py)
            import k_ftsplit
            k_ftsplit.install()
    if os.environ.get("GLM53_K_OVL") == "1":   # shared expert on a side stream, after the cache + tier wrap
        import k_overlap
        k_overlap.attach(model)
    if os.environ.get("GLM53_K_HCFUSE") == "1":   # fused hyper-connection sites, bit-exact (k_hcfuse.py)
        import k_hcfuse
        k_hcfuse.install(model)
    if os.environ.get("GLM53_LA") == "1":   # N136: decode lookahead (next forward enqueued before the token is read)
        import k_lookahead
        k_lookahead.install(model)
        k_lookahead.set_enabled(os.environ.get("GLM53_LA_ON", "1") == "1")
    if os.environ.get("GLM53_NVTX") == "1" or os.environ.get("GLM53_PROF"):   # N136 profiling hooks (n136_prof.py)
        import n136_prof
        n136_prof.install(model)
    return pools

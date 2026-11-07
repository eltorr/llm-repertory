"""How many expert slots the VRAM arena should hold after a live resize (expert_cache.Pool.resize).

No torch here, so the arithmetic runs in the unit tests. The rule is the one the arena is sized by at load
(expert_cache.Pool): the arena gets what the device-wide VRAM limit leaves after everything else in use and a
reserve for activations, capped by a slot count, in whole chunks.
"""
import math


def reserve_left(reserve_bytes, held_bytes):
    """The activation reserve still to keep free. Buffers kept since load (scratch, workspaces) came out of the reserve
    and are now counted as in use, so the reserve shrinks by them, or a regrown arena would be smaller than the one
    sized at load."""
    return max(0, reserve_bytes - max(0, held_bytes))


def target_slots(have_slots, slot_bytes, chunk_slots, capacity_slots, used_bytes=None, limit_bytes=None,
                 reserve_bytes=0, max_slots=None):
    """Slots after the resize, rounded down to whole chunks.

    have_slots      slots the arena holds now
    used_bytes      VRAM in use on the device now, every process (NVML), with the model idle
    limit_bytes     the new device-wide limit (device-wide, as NVML counts it); None: the arena can only shrink, to
                    max_slots (growing without a limit would take the activations' room)
    reserve_bytes   kept free for activations (reserve_left)
    max_slots       a cap on the slot count (GLM53_EC_MAX_SLOTS); None: none
    capacity_slots  the most the arena can index
    """
    want = have_slots
    if limit_bytes is not None:
        if used_bytes is None:
            raise ValueError("limit_bytes needs used_bytes")
        # negative room shrinks the arena; floor rounds a shortfall up to whole slots
        want = have_slots + math.floor((limit_bytes - used_bytes - reserve_bytes) / slot_bytes)
    if max_slots is not None:
        want = min(want, max_slots)
    want = max(0, min(want, capacity_slots))
    return want // chunk_slots * chunk_slots


def flex_chunks(n_chunks, elastic_bytes, chunk_bytes):
    """Indices of the chunks lent to prefill (the last ones), as at load: enough to cover elastic_bytes."""
    if elastic_bytes <= 0 or n_chunks <= 0:
        return []
    n = min(n_chunks, math.ceil(elastic_bytes / chunk_bytes))
    return list(range(n_chunks - n, n_chunks))

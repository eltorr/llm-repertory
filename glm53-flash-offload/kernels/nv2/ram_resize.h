// The RAM tier's live shrink (nv2_host.cpp ram_shrink), as plain bookkeeping with no CUDA or torch, so it can be
// tested on its own (tests/test_ram_resize.cpp).
//
// The tier's slots [0, nram) are pinned host memory in chunks; shrinking to n_keep frees the chunks above n_keep. The
// experts kept are the n_keep most recently used ones (the LRU list, newest first). Those held in a slot above n_keep
// move to a slot below it that is free once the rest have gone. Every RAM expert is also in the store on NVMe, so
// the ones that go are only dropped.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace nv2r {

struct ShrinkPlan
{
    bool ok = true;                               // false: a slot is owned but not on the LRU list (a read in flight)
    std::vector<int> evict;                       // slots whose experts are dropped (LRU-oldest first)
    std::vector<std::pair<int, int>> moves;       // (from slot >= n_keep, to slot < n_keep), kept experts
};

// owner[s] >= 0: the key in slot s; owner[s] < 0: free. Every owned slot must be on the LRU list (resident): the
// caller waits for reads in flight first. lprev: the list's links toward the oldest; lru_new: its newest slot.
inline ShrinkPlan plan_shrink(int n_keep, const std::vector<int32_t>& owner, const std::vector<int32_t>& lprev,
                              int lru_new)
{
    ShrinkPlan p;
    const int nram = int(owner.size());
    if (n_keep >= nram) return p;
    if (n_keep < 0) n_keep = 0;
    int owned = 0;
    for (int s = 0; s < nram; ++s) owned += owner[s] >= 0;
    std::vector<uint8_t> keep(size_t(nram), 0);
    int kept = 0, listed = 0;
    std::vector<int> drop_newest_first;
    for (int s = lru_new, guard = 0; s >= 0 && guard < nram; s = lprev[s], ++guard)
    {
        if (owner[s] < 0) continue;
        ++listed;
        if (kept < n_keep) { keep[s] = 1; ++kept; }
        else drop_newest_first.push_back(s);
    }
    if (listed != owned) { p.ok = false; return p; }
    p.evict.assign(drop_newest_first.rbegin(), drop_newest_first.rend());
    // free slots below n_keep once the drops are done: never owned, or owned by a dropped expert
    std::vector<int> to;
    for (int s = n_keep - 1; s >= 0; --s)
        if (owner[s] < 0 || !keep[s]) to.push_back(s);
    for (int s = n_keep; s < nram; ++s)
        if (keep[s])
        {
            p.moves.push_back({ s, to.back() });   // lowest free slot first
            to.pop_back();
        }
    return p;
}

}  // namespace nv2r

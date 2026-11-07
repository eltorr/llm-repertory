// nv2r::plan_shrink (glm53-flash-offload/kernels/nv2/ram_resize.h), the RAM tier's live shrink: built and run by
// tests/test_ram_resize.py with the system's C++ compiler. Prints "ok" and exits 0 when every check passes.
#include "ram_resize.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

struct Tier
{
    std::vector<int32_t> owner, lprev, lnext;
    int lru_old = -1, lru_new = -1;
    explicit Tier(int n) : owner(size_t(n), -1), lprev(size_t(n), -1), lnext(size_t(n), -1) {}
    void push_new(int s, int key)
    {
        owner[s] = key; lprev[s] = lru_new; lnext[s] = -1;
        if (lru_new >= 0) lnext[lru_new] = s;
        lru_new = s;
        if (lru_old < 0) lru_old = s;
    }
    std::vector<int> newest_first() const
    {
        std::vector<int> v;
        for (int s = lru_new; s >= 0; s = lprev[s]) v.push_back(owner[s]);
        return v;
    }
};

// apply the plan the way the engine does, then check the result
static void run(int nram, int resident, int n_keep, std::mt19937& rng)
{
    Tier t(nram);
    std::vector<int> slots(static_cast<size_t>(nram));
    for (int i = 0; i < nram; ++i) slots[i] = i;
    std::shuffle(slots.begin(), slots.end(), rng);
    for (int i = 0; i < resident; ++i) t.push_new(slots[i], 1000 + i);   // key 1000 + i: i-th oldest
    const std::vector<int> before = t.newest_first();
    const nv2r::ShrinkPlan p = nv2r::plan_shrink(n_keep, t.owner, t.lprev, t.lru_new);
    CHECK(p.ok);
    if (n_keep >= nram) { CHECK(p.evict.empty() && p.moves.empty()); return; }
    const int keep = std::min(resident, n_keep);
    CHECK(int(p.evict.size()) == resident - keep);
    // the dropped are the oldest, oldest first
    for (size_t i = 0; i < p.evict.size(); ++i) CHECK(t.owner[p.evict[i]] == 1000 + int(i));
    std::vector<int32_t> owner = t.owner;
    for (int s : p.evict) owner[s] = -1;
    std::set<int> dst;
    for (auto [from, to] : p.moves)
    {
        CHECK(from >= n_keep && to >= 0 && to < n_keep);
        CHECK(owner[from] >= 0 && owner[to] < 0);
        CHECK(dst.insert(to).second);
        owner[to] = owner[from]; owner[from] = -1;
    }
    std::set<int> left;
    for (int s = 0; s < nram; ++s)
        if (owner[s] >= 0) { CHECK(s < n_keep); left.insert(owner[s]); }
    CHECK(int(left.size()) == keep);
    for (int i = 0; i < keep; ++i) CHECK(left.count(before[i]) == 1);   // the newest are the ones kept
}

int main()
{
    std::mt19937 rng(7);
    run(16, 0, 8, rng);                 // empty tier
    run(16, 16, 8, rng);                // full: half go, the kept ones above 8 move down
    run(16, 5, 8, rng);                 // fewer than kept: nothing dropped
    run(16, 16, 0, rng);                // everything goes
    run(16, 16, 16, rng);               // no change
    run(16, 16, 20, rng);               // growing is not a shrink
    for (int i = 0; i < 300; ++i)
    {
        const int nram = 1 + int(rng() % 600);
        run(nram, int(rng() % (nram + 1)), int(rng() % (nram + 1)), rng);
    }
    {   // a slot owned but not on the LRU list (an NVMe read in flight): refused
        Tier t(8);
        t.push_new(1, 5); t.push_new(6, 7);
        t.owner[3] = 9;
        CHECK(!nv2r::plan_shrink(4, t.owner, t.lprev, t.lru_new).ok);
    }
    {   // exact moves on a small case: slots 0..7, LRU (old -> new) 6, 1, 7, 2; keep 2 of 4 slots
        Tier t(8);
        t.push_new(6, 60); t.push_new(1, 10); t.push_new(7, 70); t.push_new(2, 20);
        const nv2r::ShrinkPlan p = nv2r::plan_shrink(4, t.owner, t.lprev, t.lru_new);
        CHECK(p.ok && p.evict.size() == 0 && p.moves.size() == 2);
        CHECK(p.moves[0] == std::make_pair(6, 0) && p.moves[1] == std::make_pair(7, 3));
        const nv2r::ShrinkPlan q = nv2r::plan_shrink(2, t.owner, t.lprev, t.lru_new);
        CHECK(q.ok && q.evict == std::vector<int>({ 6, 1 }) && q.moves.size() == 2);
        CHECK(q.moves[0] == std::make_pair(2, 0) && q.moves[1] == std::make_pair(7, 1));
    }
    if (failures) { std::printf("%d check(s) failed\n", failures); return 1; }
    std::printf("ok\n");
    return 0;
}

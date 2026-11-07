// nv2's CPU lane kernels (kernels/nv2/ft_core.h moe_forward, ft_n135.h moe_forward4) on synthetic 3-bit mul1 experts of
// one model geometry, against the fp64 reference (ft_core.h ref_expert); built and run by test_cpu_lane.py.
//   cpu_lane_check H I EXPERTS [THREADS]      prints one line per case, then "ok" (exit 0) or "FAIL" (exit 1)
// Cases: tokens 1 / 3 / 8, top-k picks over EXPERTS experts; modes AFFINE, EXACT, I16; ft_core, n135 with its default
// unit shapes (row bands for 1-token jobs) and n135 with block units only.
#include "ft_n135.h"
#include <array>

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s H I EXPERTS [THREADS]\n", argv[0]); return 2; }
    const int H = std::atoi(argv[1]), I = std::atoi(argv[2]), E = std::atoi(argv[3]);
    const int T = argc > 4 ? std::atoi(argv[4]) : 2;
    init_perm(); init_tables();
    n135::configure_env(); n135::init_p16();

    std::mt19937_64 rng(135);
    Layer L; L.H = H; L.I = I; L.ex.resize(E);
    std::vector<std::vector<uint8_t>> tr; std::vector<std::vector<uint16_t>> sc;
    tr.reserve(3 * E); sc.reserve(6 * E);
    // random signs x (0.5 .. 1.5) x mag; the output scale carries 1/sqrt(k) as in a real checkpoint (unit-variance
    // codebook weights summed over k inputs), so activations stay well inside fp16
    auto scales = [&](int n, float mag) {
        sc.emplace_back(n);
        for (auto& s : sc.back()) s = f2h(((rng() & 1) ? 1.f : -1.f) * (0.5f + float(rng() % 1000) / 1000.f) * mag);
        return sc.back().data();
    };
    std::vector<std::array<std::vector<float>, 3>> W(E);
    for (int e = 0; e < E; ++e)
    {
        Mat* ms[3] = { &L.ex[e].g, &L.ex[e].u, &L.ex[e].d };
        for (int p = 0; p < 3; ++p)
        {
            const int k = p < 2 ? H : I, n = p < 2 ? I : H;
            tr.emplace_back(size_t(k / 16) * (n / 16) * TILE_BYTES);
            for (auto& b : tr.back()) b = uint8_t(rng());
            const uint16_t* suh = scales(k, 1.f);
            const uint16_t* svh = scales(n, 1.f / std::sqrt(float(k)));
            *ms[p] = Mat{ tr.back().data(), suh, svh, k, n };
            W[e][p] = dense_weights(*ms[p]);
        }
    }

    std::vector<int> have;
    { cpu_set_t s; sched_getaffinity(0, sizeof(s), &s); for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &s)) have.push_back(c); }
    Pool pool;
    pool.start(std::min<int>(T, int(have.size())), have);

    bool ok = true;
    const int topk = std::min(E, 4);
    for (int ntok : { 1, 3, 8 })
    {
        const std::vector<float> x = make_x(ntok, H, 1000 + ntok);
        std::vector<std::vector<std::pair<int, float>>> route(ntok);
        std::vector<float> ref(size_t(ntok) * H, 0.f), y(H);
        for (int t = 0; t < ntok; ++t)
        {
            std::vector<int> ids(E); for (int e = 0; e < E; ++e) ids[e] = e;
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int j = 0; j < topk; ++j)
            {
                const float w = 0.1f + float(rng() % 1000) / 2000.f;
                route[t].push_back({ ids[j], w });
                ref_expert(L.ex[ids[j]], W[ids[j]].data(), x.data() + size_t(t) * H, y.data());
                for (int c = 0; c < H; ++c) ref[size_t(t) * H + c] += w * y[c];
            }
        }
        double rn = 0; for (float v : ref) rn += double(v) * v;
        if (!std::isfinite(rn) || rn == 0) { std::printf("reference not finite / zero (%g)\nFAIL\n", rn); return 1; }
        std::vector<float> base(size_t(ntok) * H);
        for (int mode : { 0, 1, 2 })
        {
            std::vector<float> out(size_t(ntok) * H);
            const char* mn = mode == 0 ? "AFFINE" : mode == 1 ? "EXACT" : "I16";
            moe_forward(pool, L, x.data(), ntok, route, base.data(), mode);
            const double eb = compare(base.data(), ref.data(), base.size()).rel_rms;
            for (int variant = 0; variant < 2; ++variant)
            {
                const int keep = n135::K_.bandmaxm;
                if (variant == 1) n135::K_.bandmaxm = 0;          // block units only
                n135::moe_forward4(pool, L, x.data(), ntok, route, out.data(), mode);
                n135::K_.bandmaxm = keep;
                const double e4 = compare(out.data(), ref.data(), out.size()).rel_rms;
                const double d = compare(out.data(), base.data(), out.size()).rel_rms;
                const bool good = std::isfinite(eb) && std::isfinite(e4) && eb < 1e-2 && e4 < 1e-2 && d < 1e-2;
                ok &= good;
                std::printf("H %d I %d tokens %d %-6s ft_core rel %.2e | n135 %-6s rel %.2e, vs ft_core %.2e %s\n", H, I, ntok,
                            mn, eb, variant ? "blocks" : "bands", e4, d, good ? "" : "<-- FAIL");
            }
        }
    }
    pool.stop();
    std::printf(ok ? "ok\n" : "FAIL\n");
    return ok ? 0 : 1;
}

"""The sizing model must reproduce what was measured and refuse plans that cannot work."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engines.exl3_offload import sizing  # noqa: E402

GB = 1e9
GIB = 1024 ** 3


class SizingTests(unittest.TestCase):
    def test_matches_rtx_5090_measurements(self):
        # full mode on the 5090: ~28 GB for the model; RAM_GB 26 (3.05) / ~40 (4.05) held every expert;
        # VRAM outside the cache measured 9.0 / 9.44 GB at 262K (research/vram-scaling)
        p = sizing.plan("3.05", 28 * GB, 262144)
        self.assertTrue(p["ok"] and p["staging"])
        self.assertEqual(p["chunk"], 24576)
        self.assertAlmostEqual(p["fixed"] / GB, 9.0, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 26.6, delta=1.0)
        p = sizing.plan("4.05", 28 * GB, 262144)
        self.assertAlmostEqual(p["fixed"] / GB, 9.44, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 41.0, delta=1.5)

    def test_reserve_is_the_one_the_engine_keeps(self):
        # from 16 GB the engine keeps 1.7 GiB, set explicitly; 4.05 on the whole 5090, budget
        # 29.13 GB, loaded 7,264 slots (research/live-resize), as the plan now says
        self.assertEqual(sizing.plan("4.05", 29126885376.0, 262144)["slots"], 7264)
        lean = sizing.plan("4.05", 16 * GB, 262144)
        self.assertEqual(sizing.engine_env(lean)["GLM53_EC_RESERVE_GB"], "1.70")
        self.assertEqual(sizing.engine_env(lean)["GLM53_EC_ELASTIC_GB"], "6.0")
        self.assertEqual(lean["reserve"], sizing.ENGINE_RESERVE_GIB * GIB)
        self.assertEqual(lean["slots"], 1920)       # not 2,272: the engine had 1,824 after a live shrink
        # below 16 GB the engine gets the plan's own reserve
        small = sizing.plan("4.05", 14.5 * GB, 262144)
        self.assertAlmostEqual(float(sizing.engine_env(small)["GLM53_EC_RESERVE_GB"]) * GIB, small["reserve"],
                               delta=0.01 * GIB)

    def test_staging_holds_a_whole_layer(self):
        self.assertEqual(sizing.stage_bytes("4.05"), 2 * 512 * 2_457_600)
        self.assertGreater(512 * 2_457_600 / GIB, 1.1)        # why the 1.1 GiB default overflowed at 4.05

    def test_small_cards_get_smaller_chunks(self):
        big = sizing.plan("3.05", 22 * GB, 65536)
        small = sizing.plan("3.05", 11 * GB, 65536)
        self.assertTrue(big["ok"] and small["ok"])
        self.assertLess(small["chunk"], big["chunk"])

    def test_refuses_what_cannot_fit(self):
        p = sizing.plan("3.05", 6 * GB, 16384)
        self.assertFalse(p["ok"])
        self.assertIn("not enough", p["reason"])
        self.assertEqual(sizing.max_context("3.05", 6 * GB), 0)

    def test_8gb_class_runs_staging_free_without_a_desktop(self):
        # an 8 GB card not driving a display leaves ~8 GB: no room for staging, but the staging-free path fits at 16K
        p = sizing.plan("3.05", 8 * GB, 16384)
        self.assertTrue(p["ok"])
        self.assertFalse(p["staging"])
        self.assertGreaterEqual(p["reserve"], 0.7 * GB)
        # driving a desktop as well (~7 GB left), long prompts overran it: refuse
        self.assertFalse(sizing.plan("3.05", 7 * GB, 16384)["ok"])

    def test_baseline_presets(self):
        def b(vram, ram, display=True, platform="linux"):
            return sizing.baseline(sizing.card_bytes(vram), sizing.ram_bytes(ram), display, platform)
        # this PC: RTX 5090, 64 GB, Windows: what it runs today
        p = b(32, 64, platform="windows-wsl")
        self.assertEqual((p["quant"], p["context"]), ("4.05", 262144))
        self.assertTrue(p["measured"])
        # 24 GB: 4.05 needs the RAM to hold the rest of its experts; 3.05 below that
        self.assertEqual(b(24, 128)["quant"], "4.05")
        self.assertEqual(b(24, 32)["quant"], "3.05")
        self.assertEqual(b(16, 128)["quant"], "3.05")         # 4.05 prefill drops to 4K chunks on 16 GB
        # small cards: 64K staging-free on 12 GB, 16K staging-free on 8 GB only without a display
        p = b(12, 64)
        self.assertEqual((p["context"], p["staging"]), (65536, False))
        p = b(8, 64, display=False)
        self.assertTrue(p["ok"])
        self.assertEqual((p["context"], p["staging"]), (16384, False))
        self.assertFalse(b(8, 64)["ok"])
        # RAM: 16 GB works on Linux with smaller thresholds, not on Windows
        p = b(16, 16)
        self.assertTrue(p["ok"])
        self.assertLess(p["settings"]["ram_reserve_gb"], 6)
        self.assertFalse(b(16, 16, platform="windows-wsl")["ok"])
        # a measurement only labels a preset with a similar RAM cache
        self.assertFalse(b(16, 32)["measured"])

    def test_short_prompts_skip_staging(self):
        staged = sizing.plan("4.05", 28.5e9, 262144)
        self.assertTrue(staged["staging"])
        self.assertEqual(sizing.engine_env(staged)["GLM53_EC_STAGE_MIN"], "16384")   # 1,638 tokens x 10 experts
        free = sizing.plan("3.05", 10.5e9, 131072)
        self.assertFalse(free["staging"])
        self.assertNotIn("GLM53_EC_STAGE_MIN", sizing.engine_env(free))

    def test_ram_cache_floor_and_caps(self):
        policy = {"lean_ram_gb": 8, "ram_reserve_gb": 6, "ram_lean_gb": 6}
        need = 34 * GIB
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB), 34)
        # short of RAM: never below lean's cache (whether lean fits is the pressure policy's call)
        self.assertEqual(sizing.ram_cache_gib(need, 20 * GIB, policy, 8 * GB), 8)
        # a cap above the floor limits the cache; one below it (max_ram_cache_gib) is honoured, not raised to 8
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB, cap_gib=20), 20)
        self.assertEqual(sizing.ram_cache_gib(need, 64 * GIB, policy, 8 * GB, cap_gib=4), 4)

    def test_full_mode_headroom(self):
        self.assertAlmostEqual(sizing.headroom_bytes(32.6e9), 2.4e9)      # ran stably with ~2.1 GB free
        self.assertAlmostEqual(sizing.headroom_bytes(16.3e9), 1.63e9)
        self.assertAlmostEqual(sizing.headroom_bytes(8.5e9, display=False), 0.85e9)

    def test_doc_tables_are_generated(self):
        root = Path(__file__).resolve().parents[1]
        for doc, name, body in (("README.md", "presets", sizing.presets_table()),
                                ("README.md", "measured-summary", sizing.measured_summary()),
                                ("docs/measured.md", "measured", sizing.measured_table("../"))):
            self.assertIn(f"<!-- {name}:begin -->\n{body}\n<!-- {name}:end -->",
                          (root / doc).read_text(encoding="utf-8"),
                          f"{doc}'s {name} table is stale: "
                          f"python3 engines/exl3_offload/sizing.py presets --write {doc}")

    def test_budgeted_runs_pin_the_cache(self):
        p = sizing.plan("3.05", 16 * GB, 262144)
        self.assertIn("GLM53_EC_MAX_SLOTS", sizing.engine_env(p, cap_slots=True))
        self.assertNotIn("GLM53_EC_MAX_SLOTS", sizing.engine_env(sizing.plan("3.05", 28 * GB), cap_slots=False))

    def test_qwens_prefill_constants_are_as_measured(self):
        # the activations of the RTX 4090 runs, staging-free chunks of 1,536 (1,638 tokens x top-10 fill the pick
        # buffer) and the 6 GiB elastic share: what the derived values give for Qwen
        self.assertEqual(sizing.prefill_bytes("3.05", 24576, staging=False), 0.21e6 * 24576 + 0.3e9)
        self.assertEqual((sizing.nostage_chunk("3.05"), sizing.nostage_chunk("4.05")), (1536, 1536))
        for q in ("3.05", "4.05"):
            for budget in (16, 20, 24, 28, 29.5, 34):
                p = sizing.plan(q, budget * GB, 262144)
                self.assertEqual(sizing.engine_env(p)["GLM53_EC_ELASTIC_GB"], "6.0", (q, budget))

    def test_the_context_limit_is_the_managers(self):
        import os
        import tempfile
        if "repertory.config" not in sys.modules:                       # never a real settings dir
            os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")
        from repertory import config
        self.assertEqual(sizing.CONTEXT_LIMIT, config.MAXIMUM["context"])


class GlmSizingTests(unittest.TestCase):
    """GLM-5.3-Flash: 42 x 288 experts of 9.44 MB, top-8; its staging buffers alone are 5.4 GB."""
    Q = "glm-3.05"

    def test_its_layout(self):
        self.assertAlmostEqual(sizing.expert_bytes(self.Q) / GB, 114.15, delta=0.01)   # how-it-works.md: 114.2 GB
        self.assertEqual(sizing.stage_bytes(self.Q), 2 * 288 * 9_437_184)
        self.assertEqual(sizing.nostage_chunk(self.Q), 1792)                     # 2,048 tokens x top-8 fill it
        self.assertEqual(sizing.QUANTS[self.Q]["max_context"], 262144)         # the config's 1M, as far as we run

    def test_a_32_gb_card_at_256k(self):
        # this PC's whole-card budget: ~29.5 GB
        p = sizing.plan(self.Q, 29.5 * GB, 262144)
        self.assertTrue(p["ok"] and p["staging"])
        self.assertEqual(p["chunk"], 16384)
        self.assertAlmostEqual(p["fixed"] / GB, 8.65 + 262144 * 9504 / 1e9, delta=0.01)
        self.assertTrue(1700 <= p["slots"] <= 2000, p["slots"])                   # ~16 GB of experts, 14-16%
        # the engine may lend each prefill its staging buffers and the chunk's activations, beyond the reserve
        env = sizing.engine_env(p)
        lent = float(env["GLM53_EC_ELASTIC_GB"]) * GIB
        self.assertGreaterEqual(lent + float(env["GLM53_EC_RESERVE_GB"]) * GIB, sizing.prefill_bytes(self.Q, 16384))
        self.assertLessEqual(lent, p["cache"])                                    # and the cache has that much
        self.assertEqual(sizing.live_resize_args(p)["elastic_bytes"], int(lent))
        # the RAM cache to hold 90% of the experts with VRAM: ~80 GiB, more than this PC has
        self.assertAlmostEqual(p["ram_target"] / GIB, 79, delta=2)

    def test_smaller_cards(self):
        self.assertEqual(sizing.plan(self.Q, 22 * GB, 65536)["chunk"], 8192)
        p = sizing.plan(self.Q, 16 * GB, 65536)                                  # no room for 5.4 GB of staging
        self.assertTrue(p["ok"])
        self.assertEqual((p["staging"], p["chunk"]), (False, 1792))
        self.assertFalse(sizing.plan(self.Q, 10 * GB, 65536)["ok"])
        # staged prefill at 256K: ~19 GB, 6.5 GB more than Qwen's 3.05 (its staging and its weights)
        self.assertAlmostEqual(sizing.min_budget(self.Q, 262144) / GB, 18.96, delta=0.01)
        self.assertGreater(sizing.min_budget(self.Q, 262144), sizing.min_budget("3.05", 262144) + 6 * GB)

    def test_it_fits_this_pc_with_half_its_experts_in_memory(self):
        f = sizing.fit(self.Q, 29.5 * GB, sizing.ram_bytes(64), "windows-wsl")
        self.assertTrue(f["ram_fits"])
        self.assertEqual((f["context"], f["chunk"]), (262144, 16384))
        self.assertTrue(0.4 < f["in_memory"] < 0.55, f["in_memory"])
        self.assertGreater(sizing.fit(self.Q, 29.5 * GB, sizing.ram_bytes(128), "linux")["in_memory"], 0.89)

    def test_the_presets_stay_qwens(self):
        for vram, ram in ((8, 16), (16, 64), (32, 64), (48, 128)):
            b = sizing.baseline(sizing.card_bytes(vram), sizing.ram_bytes(ram))
            self.assertIn(b.get("quant", "3.05"), ("3.05", "4.05"))


if __name__ == "__main__":
    unittest.main()

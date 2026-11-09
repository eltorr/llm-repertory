"""Setup's model steps come from the catalog: one per model, with its label, sizes, paths and pin; which are shown
follows curation; the scripts take the catalog's values (and still work the way they were called before)."""
import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from repertory import catalog, config, hardware  # noqa: E402
from repertory import setup as setup_mod  # noqa: E402

DATA = json.loads(catalog.CATALOG.read_text(encoding="utf-8"))
QWEN = "turboderp/Qwen3.8-Flash-Next-exl3"
# bash here must be a real Linux (or macOS) shell: on Windows `bash` can be WSL's, and these tests never run in WSL
POSIX_BASH = sys.platform != "win32" and shutil.which("bash")


def with_curation(**states):
    """A catalog whose models (by quant) carry these curation states, and the others none (untested), whatever the
    real catalog's results are."""
    data = copy.deepcopy(DATA)
    for m in data["models"]:
        m.pop("curation", None)
        if m["quant"] in states:
            m["curation"] = {"status": states[m["quant"]], "date": "2026-11-07", "checks": {}}
    return catalog.Catalog(data)


def setup_for(cfg=None):
    """Setup as on Linux."""
    svc = types.SimpleNamespace(host=types.SimpleNamespace(kind="linux"), cfg=dict(config.DEFAULTS, **(cfg or {})))
    return setup_mod.Setup(svc)


@mock.patch.object(setup_mod, "IS_WINDOWS", False)
class ModelStepTests(unittest.TestCase):
    def test_one_step_per_model_with_its_label_sizes_paths_and_pin(self):
        patcher = mock.patch.object(catalog, "load", with_curation)      # untested, whatever the real results
        patcher.start()
        self.addCleanup(patcher.stop)
        steps = {s["id"]: s for s in setup_mod.model_steps()}
        self.assertEqual(list(steps), ["m305", "m405", "mglm305", "mglm205"])
        s = steps["m305"]
        self.assertEqual(s["title"], "Model: 3.05 bpw (fastest)")
        self.assertEqual(s["desc"], "~85 GB download + ~47 GB expert store. Resumable.")
        self.assertEqual(s["args"], ["3.05", "--dir", "model", "--store", "store/qwen.bin", "--repo", QWEN,
                                     "--revision", "69e33439ae950f17bcbe95c98f117d80f759ab6d",
                                     "--branch", "3.05bpw_h5_ng5"])
        self.assertEqual(steps["m405"]["title"], "Model: 4.05 bpw (higher fidelity)")
        self.assertEqual(steps["m405"]["desc"], "~107 GB download + ~62 GB expert store. Resumable.")
        self.assertEqual(steps["m405"]["args"][1:5], ["--dir", "model-4.05", "--store", "store/qwen405.bin"])
        self.assertEqual((s["model"], s["curation"], s["offered"]), ("qwen3.8-flash-next-exl3-3.05", "untested", True))
        glm = steps["mglm305"]
        self.assertEqual((glm["title"], glm["desc"]),
                         ("Model: GLM-5.3-Flash 3.05 bpw", "~125.3 GB download + ~117.3 GB expert store. Resumable."))
        self.assertEqual(glm["args"], ["glm-3.05", "--dir", "model-glm-3.05", "--store", "store/glm305.bin", "--repo",
                                       "turboderp/GLM-5.3-Flash-exl3", "--revision",
                                       "332ab457b709b7ba30dd9a448be5de03b80a7ac9", "--branch", "3.05bpw"])
        self.assertEqual((glm["model"], glm["quant"], glm["curation"], glm["offered"]),
                         ("glm-5.3-flash-exl3-3.05", "glm-3.05", "untested", False))
        self.assertEqual([x["id"] for x in setup_for().steps],
                         ["driver", "base", "engine", "build", "m305", "m405"])          # Linux; GLM isn't curated
        self.assertEqual([x["id"] for x in setup_for({"offer_uncurated": True}).steps][-2:],
                         ["mglm305", "mglm205"])
        self.assertEqual([x["id"] for x in setup_for({"quant": "glm-3.05"}).steps][-1], "mglm305")   # the chosen one
        self.assertEqual(setup_for().step("mglm305")["args"][0], "glm-3.05")          # runnable by name either way

    def test_check_sh_is_told_where_each_model_lives(self):
        self.assertEqual(setup_mod.model_specs(), ["m305=model:store/qwen.bin", "m405=model-4.05:store/qwen405.bin",
                                                   "mglm305=model-glm-3.05:store/glm305.bin",
                                                   "mglm205=model-glm-2.05:store/glm205.bin"])

    def test_a_model_that_failed_curation_is_hidden_unless_installed_chosen_or_asked_for(self):
        cat = with_curation(**{"3.05": "curated", "4.05": "failed"})
        with mock.patch.object(setup_mod.catalog, "load", lambda: cat):
            ids = [s["id"] for s in setup_for({"quant": "3.05"}).steps]
            self.assertIn("m305", ids)
            self.assertNotIn("m405", ids)
            self.assertIn("m405", [s["id"] for s in setup_for({"quant": "4.05"}).steps])       # the chosen model
            self.assertIn("m405", [s["id"] for s in setup_for({"offer_uncurated": True}).steps])
            installed = setup_for({"quant": "3.05"})
            installed.status["m405"] = "ok"
            self.assertIn("m405", [s["id"] for s in installed.steps])
            self.assertEqual(setup_for().step("m405")["args"][0], "4.05")       # still runnable by name (the CLI)
            report = {s["id"]: s for s in installed.report()["steps"]}
            self.assertEqual((report["m405"]["curation"], report["m405"]["status"]), ("failed", "ok"))

    def test_statuses_follow_what_check_sh_reports_for_each_model(self):
        st = setup_for().statuses({"gpus": [{}], "driver": {}, "repo": {},
                                   "engine": {"cuda": "1", "driver": "1", "engine": "1", "env": "1", "build": "1",
                                              "build_src": "1", "m305": "1", "m405": "0"}})
        self.assertEqual((st["m305"], st["m405"]), ("ok", "missing"))
        st = setup_for().statuses({"gpus": [{}], "driver": {}, "engine": {"error": "no WSL"}})
        self.assertEqual((st["m305"], st["m405"]), ("unknown", "unknown"))
        self.assertEqual(hardware.installed({"m305": "1"}), {"3.05": True, "4.05": False, "glm-3.05": False,
                                                                 "glm-2.05": False})
        self.assertEqual(hardware.installed({"mglm305": "1"})["glm-3.05"], True)

    def test_running_a_model_step_passes_the_catalogs_values_and_the_legacy_folder(self):
        ran = []

        class Host:
            kind, engine_dir, cfg = "linux", "/srv/engine", {"legacy_engine_dir": "/srv/old"}

            def to_engine_path(self, p):
                return "/repo/setup/" + Path(p).name

            def engine_argv(self, *a):
                return list(a)

            async def stream(self, argv, emit):
                ran.append(argv)
                return 0
        svc = types.SimpleNamespace(host=Host(), cfg={}, observe=False, be=types.SimpleNamespace(state="stopped"),
                                    event=lambda *a: None)
        s = setup_mod.Setup(svc)

        async def no_check():
            return None
        s.check = no_check
        asyncio.run(s._run(s.step("m405")))
        self.assertEqual(ran, [["bash", "/repo/setup/50_model.sh", "/srv/engine", "4.05", "--dir", "model-4.05",
                                "--store", "store/qwen405.bin", "--repo", QWEN, "--revision",
                                "55a732e0c4c3d4614bc42b68493bb930d9b02c0a", "--branch", "4.05bpw_h6_ng6", "/srv/old"]])


class DashboardTests(unittest.TestCase):
    def service(self, **cfg):
        from repertory.service import Service
        svc = types.SimpleNamespace(cfg=dict(config.DEFAULTS, **cfg), setup=types.SimpleNamespace(hw=None))
        for name in ("model", "client_models", "model_object"):
            setattr(svc, name, getattr(Service, name).__get__(svc))
        return svc, Service

    def test_the_model_choices_and_the_served_name_come_from_the_catalog(self):
        svc, Service = self.service(quant="4.05")
        with mock.patch.object(catalog, "load", with_curation):          # untested, whatever the real results
            choices = Service.model_choices(svc)
        self.assertEqual([(c["quant"], c["label"], c["curation"]) for c in choices],
                         [("3.05", "3.05 bpw (fastest)", "untested"),
                          ("4.05", "4.05 bpw (higher fidelity)", "untested")])
        self.assertEqual(Service.models_list(svc)["data"][0]["id"], "qwen3.8-flash-next-exl3-4.05")
        cat = with_curation(**{"3.05": "failed", "4.05": "curated"})
        with mock.patch.object(catalog, "load", lambda: cat):
            self.assertEqual([c["quant"] for c in Service.model_choices(svc)], ["4.05"])        # failed: not offered
            svc.cfg["quant"] = "3.05"
            self.assertEqual([c["quant"] for c in Service.model_choices(svc)], ["3.05", "4.05"])  # it's the chosen one
            svc.cfg.update(quant="4.05", offer_uncurated=True)
            self.assertEqual([c["offered"] for c in Service.model_choices(svc)], [False, True, False, False])

    def test_the_baseline_keeps_glm_and_its_context(self):
        # the presets are Qwen's: on GLM, applying the baseline takes its RAM thresholds, not Qwen's model
        from engines.exl3_offload import sizing
        b = sizing.baseline(sizing.card_bytes(32), sizing.ram_bytes(64), True, "windows-wsl")
        cfg = dict(config.DEFAULTS, quant="glm-3.05", context=131072, ram_reserve_gb=4.0)
        level, title, detail, settings = hardware.baseline_item(b, cfg)
        self.assertEqual((level, settings), ("info", {"ram_reserve_gb": 6.0}))
        self.assertIn("The presets are qwen3.8-flash-next's: glm-3.05 keeps its model and context.", detail)
        _, _, _, settings = hardware.baseline_item(b, dict(cfg, quant="3.05"))
        self.assertEqual((settings["quant"], settings["context"]), ("4.05", 262144))   # Qwen: as before

    def test_glm_is_offered_once_curated_and_serves_under_its_own_name(self):
        svc, Service = self.service(quant="4.05")
        with mock.patch.object(catalog, "load", with_curation):
            self.assertNotIn("glm-3.05", [c["quant"] for c in Service.model_choices(svc)])   # untested
            svc.cfg["quant"] = "glm-3.05"                                                    # the chosen one
            choices = {c["quant"]: c for c in Service.model_choices(svc)}
            self.assertEqual((choices["glm-3.05"]["label"], choices["glm-3.05"]["offered"]),
                             ("GLM-5.3-Flash 3.05 bpw", False))
            self.assertEqual(Service.models_list(svc)["data"][0]["id"], "glm-5.3-flash-exl3-3.05")
        cat = with_curation(**{"glm-3.05": "curated"})
        with mock.patch.object(catalog, "load", lambda: cat):
            svc.cfg["quant"] = "3.05"
            self.assertIn("glm-3.05", [c["quant"] for c in Service.model_choices(svc)])


@unittest.skipUnless(POSIX_BASH, "needs a Linux shell")
class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="repertory-engine-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def fake_model(self, root, folder, store):
        (root / folder).mkdir(parents=True)
        (root / folder / "config.json").write_text("{}")
        (root / store).parent.mkdir(parents=True, exist_ok=True)
        (root / store).write_bytes(b"x")
        (root / store).with_suffix(".json").write_text("{}")

    def test_check_sh_reports_the_models_it_is_given(self):
        self.fake_model(self.root, "model-x", "store/x.bin")
        (self.root / "model-y").mkdir()
        (self.root / "model-y" / "config.json").write_text("{}")
        (self.root / "model-y" / "weights.safetensors.part").write_bytes(b"")      # still downloading
        out = subprocess.run(["bash", str(REPO / "setup" / "check.sh"), str(self.root), "mx=model-x:store/x.bin",
                              "my=model-y:store/y.bin", "bad=../x:store/x.bin"],
                             capture_output=True, text=True, timeout=120).stdout
        vals = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        self.assertEqual((vals.get("mx"), vals.get("my")), ("1", "0"))
        self.assertNotIn("bad", vals)                                       # not a path inside the checkout
        self.assertNotIn("m305", vals)                                      # only the models it was given

    def test_check_sh_called_the_old_way_reports_qwens_two(self):
        self.fake_model(self.root, "model", "store/qwen.bin")
        out = subprocess.run(["bash", str(REPO / "setup" / "check.sh"), str(self.root)],
                             capture_output=True, text=True, timeout=120).stdout
        vals = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        self.assertEqual((vals.get("m305"), vals.get("m405")), ("1", "0"))

    def link_run(self, *args):
        """50_model.sh with a legacy checkout that has the model: it links it and stops before downloading."""
        old = self.root / "old"
        engine = self.root / "engine"
        engine.mkdir()
        return old, engine, subprocess.run(["bash", str(REPO / "setup" / "50_model.sh"), str(engine), *args],
                                           capture_output=True, text=True, timeout=120)

    def test_50_model_sh_takes_the_catalogs_paths(self):
        self.fake_model(self.root / "old", "model-z", "store/z.bin")
        old, engine, r = self.link_run("9.9", "--dir", "model-z", "--store", "store/z.bin", "--repo", "a/b",
                                       "--revision", "0" * 40, str(self.root / "old"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((engine / "model-z" / "config.json").is_file())
        self.assertTrue((engine / "store" / "z.bin").is_file())

    def test_50_model_sh_called_the_old_way_still_knows_qwen(self):
        self.fake_model(self.root / "old", "model-4.05", "store/qwen405.bin")
        old, engine, r = self.link_run("4.05", str(self.root / "old"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((engine / "store" / "qwen405.bin").is_file())

    def test_50_model_sh_refuses_paths_outside_the_checkout_and_unknown_quants(self):
        _, _, r = self.link_run("9.9", "--dir", "../escape", "--store", "store/z.bin")
        self.assertEqual(r.returncode, 2)
        self.assertIn("outside the engine checkout", r.stdout)
        shutil.rmtree(self.root / "engine")
        _, _, r = self.link_run("9.9")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown quant", r.stdout)


if __name__ == "__main__":
    unittest.main()

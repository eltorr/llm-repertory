"""The first start after QwenFlash (repertory/config.py import_legacy): its settings and learned limits come along
once. Each case runs in a child process, as the settings folder is fixed when the config module loads."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOAD = ("import json, sys; from repertory import config; cfg = config.load_config(); "
        "print(json.dumps({'cfg': cfg, 'learned': config.load_json('learned.json', None)}))")


def start(home, legacy=None):
    env = {k: v for k, v in os.environ.items() if k not in ("QWENFLASH_HOME", "REPERTORY_HOME")}
    env["REPERTORY_HOME"] = str(home)
    if legacy is not None:
        env["QWENFLASH_HOME"] = str(legacy)
    r = subprocess.run([sys.executable, "-B", "-c", LOAD], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    if r.returncode:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout.splitlines()[-1])


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.home, self.legacy = base / "Repertory", base / "QwenFlash"
        self.legacy.mkdir()
        (self.legacy / "config.json").write_text(json.dumps({"dashboard_token": "kept-token", "quant": "4.05",
                                                              "network_port": 8080, "repo_dir": ""}))
        (self.legacy / "learned.json").write_text(json.dumps({"GPU|4.05|262144": {"vram_cap": 2.0e10}}))
        (self.legacy / "bench" / "20261105-224719").mkdir(parents=True)
        (self.legacy / "bench" / "20261105-224719" / "results.jsonl").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_settings_token_and_learned_limits_carry_over(self):
        out = start(self.home, self.legacy)
        cfg = out["cfg"]
        self.assertEqual((cfg["dashboard_token"], cfg["quant"], cfg["network_port"]), ("kept-token", "4.05", 8080))
        self.assertEqual(out["learned"], {"GPU|4.05|262144": {"vram_cap": 2.0e10}})
        self.assertEqual(cfg["listen_port"], 30000)                           # the ports stay as they were
        # Windows: the engine checkout inside WSL stays where QwenFlash had it, with its models
        self.assertEqual(cfg["repo_dir"], "")      # Repertory's own engine checkout, never QwenFlash's
        self.assertEqual(cfg["legacy_engine_dir"], "/root/qwen-flash" if sys.platform == "win32" else "")
        self.assertTrue((self.home / "config.json").exists())
        self.assertTrue((self.home / "bench" / "20261105-224719" / "results.jsonl").exists())   # its benchmark runs

    def test_only_once(self):
        start(self.home, self.legacy)
        (self.legacy / "config.json").write_text(json.dumps({"quant": "3.05", "dashboard_token": "other"}))
        self.assertEqual(start(self.home, self.legacy)["cfg"]["dashboard_token"], "kept-token")

    def test_a_repertory_config_is_never_replaced(self):
        self.home.mkdir()
        (self.home / "config.json").write_text(json.dumps({"dashboard_token": "mine"}))
        out = start(self.home, self.legacy)
        self.assertEqual(out["cfg"]["dashboard_token"], "mine")
        self.assertIsNone(out["learned"])

    def test_unreadable_or_missing_settings_start_from_the_defaults(self):
        (self.legacy / "config.json").write_text("{not json")
        self.assertEqual(start(self.home, self.legacy)["cfg"]["quant"], "3.05")
        empty = Path(self.tmp.name) / "none"
        self.assertEqual(start(Path(self.tmp.name) / "other", empty)["cfg"]["dashboard_token"], "")

    def test_a_settings_folder_of_its_own_imports_nothing_by_itself(self):
        # REPERTORY_HOME without QWENFLASH_HOME (the tests, a second install): QwenFlash's default folder is left alone
        self.assertEqual(start(self.home)["cfg"]["dashboard_token"], "")


if __name__ == "__main__":
    unittest.main()

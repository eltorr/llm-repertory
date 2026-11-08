"""The catalog (repertory/catalog.py): it validates against its schema, the validator catches what the schema forbids,
the lookups, what is offered, and what reads it (sizing's per-model numbers)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from engines.exl3_offload import sizing  # noqa: E402
from repertory import catalog  # noqa: E402

SCHEMA = json.loads(catalog.SCHEMA.read_text(encoding="utf-8"))
DATA = json.loads(catalog.CATALOG.read_text(encoding="utf-8"))


def errors_for(change):
    """The catalog's errors after change(data) on a copy."""
    data = copy.deepcopy(DATA)
    change(data)
    return catalog.check(data, SCHEMA)


class CatalogFileTests(unittest.TestCase):
    def test_the_catalog_validates(self):
        self.assertEqual(catalog.check(DATA, SCHEMA), [])
        cat = catalog.load()
        self.assertEqual([m["quant"] for m in cat.models_for("exl3-offload")], ["3.05", "4.05", "glm-3.05", "glm-2.05"])

    def test_the_file_is_kept_as_json_dumps_writes_it(self):
        # save_curation rewrites the file: it must come back byte for byte, so only the curation changes in a diff
        text = catalog.CATALOG.read_text(encoding="utf-8")
        self.assertEqual(json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n", text)

    def test_prepare_steps_match_the_paths_and_pins(self):
        # setup runs setup/50_model.sh with the paths and pins; the prepare steps document the same commands
        for m in DATA["models"]:
            download, pack = (s["run"] for s in m["prepare"])
            self.assertEqual(download[download.index("--model-dir") + 1], "./" + m["paths"]["model_dir"], m["id"])
            self.assertEqual(download[download.index("--branch") + 1], m["source"]["branch"], m["id"])
            for flag, value in (("--repo", m["source"]["huggingface"]), ("--sha", m["source"]["revision"])):
                if flag in download:
                    self.assertEqual(download[download.index(flag) + 1], value, m["id"])
            self.assertEqual(pack[pack.index("--name") + 1], catalog.store_name(m), m["id"])
            self.assertEqual(pack[pack.index("--model") + 1], "./" + m["paths"]["model_dir"], m["id"])
            self.assertEqual(m["prepare"][1]["creates"], m["paths"]["store"], m["id"])

    def test_the_download_scripts_pins_are_the_catalogs(self):
        # download.py keeps Qwen's pins for a run by hand with only --branch; another repository's entry passes its
        # pin as --repo and --sha
        from engines.exl3_offload import download
        for m in DATA["models"]:
            if m["source"]["huggingface"] == download.REPO:
                self.assertEqual(download.PINNED[m["source"]["branch"]], m["source"]["revision"], m["id"])
            else:
                run = m["prepare"][0]["run"]
                self.assertIn("--sha", run, m["id"])
                self.assertIn("--repo", run, m["id"])
                self.assertNotIn(m["source"]["branch"], download.PINNED, m["id"])


class ValidatorTests(unittest.TestCase):
    def test_schema_violations(self):
        cases = {
            "missing": (lambda d: d["models"][0].pop("label"), "label is missing"),
            "type": (lambda d: d["models"][0].update(bpw="3.05"), "should be number"),
            "pattern": (lambda d: d["models"][0]["source"].update(revision="main"), "doesn't match"),
            "enum": (lambda d: d["models"][0].update(format="onnx"), "should be one of"),
            "extra": (lambda d: d["models"][0].update(colour="blue"), "colour is not allowed"),
            "const": (lambda d: d.update(schema_version=2), "should be 1"),
            "minimum": (lambda d: d["models"][0].update(bpw=0), "should be more than 0"),
            "min items": (lambda d: d["models"][0].update(engines=[]), "at least 1"),
            "one of": (lambda d: d["engines"][0]["source"].update(git="https://example.invalid/x.git"),
                       "exactly one of 2 forms"),
            "ref": (lambda d: d["models"][0].update(id="Not An Id"), "doesn't match"),
            "integer": (lambda d: d["models"][0]["facts"].update(moe_layers=48.5), "should be integer"),
            "bool is no integer": (lambda d: d["models"][0]["facts"].update(moe_layers=True), "should be integer"),
            "date": (lambda d: d["models"][0]["measured"][0].update(date="2026-13-40"), "is not a date"),
            "curation status": (lambda d: d["models"][0].update(curation={"status": "great", "date": "2026-11-07",
                                                                          "checks": {}}), "should be one of"),
        }
        for name, (change, message) in cases.items():
            errs = errors_for(change)
            self.assertTrue(any(message in e for e in errs), f"{name}: {errs}")

    def test_paths_say_where(self):
        errs = errors_for(lambda d: d["models"][1]["source"].update(revision="xyz"))
        self.assertTrue(errs[0].startswith("$.models[1].source.revision: "), errs)

    def test_what_the_schema_cant_express(self):
        cases = {
            "unknown engine": (lambda d: d["models"][0].update(engines=["vllm"]), "engine vllm is not in the catalog"),
            "duplicate id": (lambda d: d["models"][1].update(id=d["models"][0]["id"]), "is used twice"),
            "duplicate quant": (lambda d: d["models"][1].update(quant="3.05"), "quant 3.05 is used twice"),
            "format": (lambda d: d["models"][0].update(format="gguf"), "doesn't read the gguf format"),
            "path outside": (lambda d: d["models"][0]["paths"].update(model_dir="../model"), "relative path"),
            "absolute path": (lambda d: d["models"][0]["paths"].update(store="/root/store/q.bin"), "relative path"),
            "placeholder": (lambda d: d["engines"][0]["serve"]["placeholders"].remove("port"),
                            "placeholders used but not listed: ['port']"),
        }
        for name, (change, message) in cases.items():
            errs = errors_for(change)
            self.assertTrue(any(message in e for e in errs), f"{name}: {errs}")

    def test_a_keyword_the_validator_doesnt_know_is_an_error_not_ignored(self):
        with self.assertRaises(catalog.CatalogError):
            catalog.validate({"a": 1}, {"type": "object", "maxProperties": 0})

    def test_an_invalid_catalog_doesnt_load(self):
        with tempfile.TemporaryDirectory() as d:
            bad = copy.deepcopy(DATA)
            bad["models"][0].pop("facts")
            p = Path(d) / "catalog.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(catalog.CatalogError) as e:
                catalog.load(p)
            self.assertIn("facts is missing", str(e.exception))


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.cat = catalog.load()

    def test_models_by_id_or_quant_and_their_engine(self):
        m = self.cat.model("qwen3.8-flash-next-exl3-4.05")
        self.assertIs(self.cat.model("4.05"), m)
        self.assertEqual(self.cat.engine_of(m)["id"], "exl3-offload")
        self.assertEqual(m["paths"], {"model_dir": "model-4.05", "store": "store/qwen405.bin"})
        self.assertEqual(m["source"]["revision"], "55a732e0c4c3d4614bc42b68493bb930d9b02c0a")
        self.assertEqual(catalog.setup_id(m), "m405")
        self.assertEqual(catalog.store_name(m), "qwen405")
        self.assertEqual(catalog.served_name(m), "qwen3.8-flash-next")
        self.assertIsNone(self.cat.find("5.0"))
        with self.assertRaises(KeyError):
            self.cat.model("5.0")
        with self.assertRaises(KeyError):
            self.cat.engine("vllm")

    def test_glm(self):
        m = self.cat.model("glm-5.3-flash-exl3-3.05")
        self.assertIs(self.cat.model("glm-3.05"), m)
        self.assertEqual((m["family"], m["label"], m["bpw"]), ("glm-5.3-flash", "GLM-5.3-Flash 3.05 bpw", 3.05))
        self.assertEqual(m["source"], {"huggingface": "turboderp/GLM-5.3-Flash-exl3", "branch": "3.05bpw",
                                       "revision": "332ab457b709b7ba30dd9a448be5de03b80a7ac9", "verify": "hub-hashes"})
        self.assertEqual(m["paths"], {"model_dir": "model-glm-3.05", "store": "store/glm305.bin"})
        self.assertEqual((catalog.setup_id(m), catalog.store_name(m), catalog.served_name(m)),
                         ("mglm305", "glm305", "glm-5.3-flash"))
        self.assertIn("Glm5NextForConditionalGeneration", self.cat.engine_of(m)["supports"]["architectures"])
        # offered once curated, not before
        self.assertFalse(m["offer_untested"])
        self.assertFalse(catalog.offered(dict(m, curation={"status": "untested", "date": "2026-11-07", "checks": {}})))
        self.assertTrue(catalog.offered(dict(m, curation={"status": "curated", "date": "2026-11-07", "checks": {}})))

    def test_glms_facts_follow_from_its_config_and_the_engines_layout(self):
        # config.json (text_config): 45 layers, the first 3 dense (42 MoE), 288 routed experts, top-8, one MTP layer;
        # hidden 4096, expert width 2048
        f = self.cat.model("glm-3.05")["facts"]
        self.assertEqual((f["experts_per_layer"], f["moe_layers"], f["experts_per_token"], f["mtp"]),
                         (288, 42, 8, True))
        # a VRAM slot is the gate, up and down trellises at 3 bits; a store record adds the scales, 4K-aligned
        # (glm53-flash-offload/docs/how-it-works.md: 9,437,184 and 9,474,048 B)
        self.assertEqual(f["expert_bytes"], 3 * 4096 * 2048 * 3 // 8)
        self.assertEqual((f["store_record_bytes"], f["store_record_bytes"] % 4096), (9_474_048, 0))
        # what the packer wrote: the 42 layers and the MTP layer, 117.33 GB
        self.assertAlmostEqual(f["store_record_bytes"] * 288 * 43 / 1e9, 117.33, delta=0.01)
        # KV cache per token: 11 sparse-attention layers of exllamav3's MLA cache (cache/mla.py): the 512-wide latent
        # (fp16, or quantized in 32-value groups with an fp16 scale each), the indexer's key plane (2 x 128 fp16)
        # and its pooled keys (128 fp16 per 4 tokens); no RoPE part (qk_rope_head_dim 0)
        planes = 2 * 128 * 2 + 128 // 4 * 2
        want = {"16": 11 * (512 * 2 + planes)}
        want.update({str(b): 11 * (512 // 32 * b * 4 + 512 // 32 * 2 + planes) for b in (2, 4, 8)})
        self.assertEqual(f["kv_bytes_per_token"], want)
        self.assertEqual(f["max_context"], 1048576)                     # max_position_embeddings

    def test_reference_paths_stay_in_the_reference_folder(self):
        self.assertEqual(catalog.reference_path("qwen3.8-flash-next-exl3-3.05").parent, catalog.REFERENCE_DIR)
        for bad in ("../catalog", "a/b", "", "X"):
            with self.assertRaises(ValueError, msg=bad):
                catalog.reference_path(bad)


class OfferTests(unittest.TestCase):
    def test_what_is_offered(self):
        m = copy.deepcopy(DATA["models"][0])
        m.pop("curation", None)
        self.assertEqual(catalog.status(m), "untested")
        self.assertTrue(m["offer_untested"])                 # Qwen's two predate curation: offered, as untested
        self.assertTrue(catalog.offered(m))
        for st, shown in (("curated", True), ("failed", False), ("untested", True)):
            m["curation"] = {"status": st, "date": "2026-11-07", "checks": {}}
            self.assertEqual(catalog.offered(m), shown, st)
            self.assertTrue(catalog.offered(m, show_all=True), st)
        m.pop("offer_untested")                              # a new entry: only once it is curated
        for st, shown in (("curated", True), ("failed", False), ("untested", False)):
            m["curation"] = {"status": st, "date": "2026-11-07", "checks": {}}
            self.assertEqual(catalog.offered(m), shown, st)
        m.pop("curation")
        self.assertFalse(catalog.offered(m))
        self.assertTrue(catalog.offered(m, show_all=True))


class SaveTests(unittest.TestCase):
    def test_a_curation_result_is_written_into_its_entry_and_nothing_else_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "catalog.json"
            p.write_text(catalog.CATALOG.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            cur = {"status": "untested", "date": "2026-11-07", "engine_commit": "abc",
                   "machine": {"gpu": "Test GPU", "vram_gb": 32.0, "ram_gb": 64.0, "platform": "linux"},
                   "checks": {"compat": {"ok": True, "date": "2026-11-07", "items": [{"name": "entry", "ok": True}]}}}
            catalog.save_curation("qwen3.8-flash-next-exl3-3.05", cur, p)
            got = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(got["models"][0]["curation"], cur)
            want = copy.deepcopy(DATA)                                      # the entry's record replaced, and only it
            want["models"][0]["curation"] = cur
            self.assertEqual(got, want)
            self.assertEqual(catalog.status(catalog.load(p).model("3.05")), "untested")
            with self.assertRaises(catalog.CatalogError):                  # checked before it is written
                catalog.save_curation("qwen3.8-flash-next-exl3-3.05", {"status": "fine"}, p)
            with self.assertRaises(KeyError):
                catalog.save_curation("nope", cur, p)

    def test_flat_puts_each_list_of_numbers_on_one_line(self):
        data = {"top": [[1, 2.5, -3e-05], [4]], "text": "x = [0,1]\nnext", "none": [], "objects": [{"k": 1}]}
        text = catalog.flat_numbers(json.dumps(data, indent=1))
        self.assertEqual(json.loads(text), data)
        self.assertIn('"top": [\n  [1, 2.5, -3e-05],\n  [4]\n ]', text)
        self.assertIn('"x = [0,1]\\nnext"', text)                     # a string is never touched
        self.assertIn('"objects": [\n  {', text)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ref.json"
            catalog.write_json(p, data, flat=True)
            self.assertEqual(p.read_bytes(), (text + "\n").encode())


class SizingTests(unittest.TestCase):
    def test_sizings_numbers_are_the_catalogs(self):
        for m in DATA["models"]:
            q, f = sizing.QUANTS[m["quant"]], m["facts"]
            self.assertEqual((q["rec"], q["fixed0"]), (f["expert_bytes"], f["fixed_vram_bytes"]))
            self.assertEqual((q["act"], q["topk"]), (f["prefill_act_bytes_per_token"], f["experts_per_token"]))
            self.assertEqual(q["max_context"], min(f["max_context"], sizing.CONTEXT_LIMIT))
            self.assertEqual(sizing.prefill_bytes(m["quant"], 1000, staging=False) - sizing.PREFILL_EXTRA,
                             1000 * f["prefill_act_bytes_per_token"])
            self.assertEqual(sizing.expert_bytes(m["quant"]),
                             f["expert_bytes"] * f["experts_per_layer"] * f["moe_layers"])
            self.assertEqual(sizing.fixed_bytes(m["quant"], 1000, 8) - q["fixed0"],
                             1000 * f["kv_bytes_per_token"]["8"])
            self.assertEqual((q["model_dir"], q["store"], q["label"]),
                             (m["paths"]["model_dir"], m["paths"]["store"], m["label"]))
        # the calibration the tests in test_sizing.py check
        self.assertEqual(sizing.QUANTS["3.05"]["fixed0"], 6.15 * 1e9)
        self.assertEqual(sizing.QUANTS["4.05"]["fixed0"], 6.62 * 1e9)
        self.assertEqual({sizing.QUANTS[q]["act"] for q in ("3.05", "4.05")}, {0.21e6})    # the RTX 4090 runs


if __name__ == "__main__":
    unittest.main()

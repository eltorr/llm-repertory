"""/v1/models and the model a request names (Service.client_models, resolve_model, route): the installed catalog
models in OpenAI's form, the loaded one first; names by id, quant or family; and a request for another installed
model switching to it, after the requests already running, one switch at a time."""
import asyncio
import calendar
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repertory import catalog, config  # noqa: E402
from repertory import service as service_mod  # noqa: E402
from repertory.service import Service  # noqa: E402

DATA = json.loads(catalog.CATALOG.read_text(encoding="utf-8"))
Q305, Q405 = "qwen3.8-flash-next-exl3-3.05", "qwen3.8-flash-next-exl3-4.05"
G305, G205 = "glm-5.3-flash-exl3-3.05", "glm-5.3-flash-exl3-2.05"
DAY = calendar.timegm((2026, 11, 8, 0, 0, 0))


def cat_with(**states):
    """The catalog with these curation states (by quant), the others untested and not offered."""
    data = copy.deepcopy(DATA)
    for m in data["models"]:
        m.pop("curation", None)
        m.pop("offer_untested", None)
        if m["quant"] in states:
            m["curation"] = {"status": states[m["quant"]], "date": "2026-11-08", "checks": {}}
    return catalog.Catalog(data)


ALL = {"3.05": "curated", "4.05": "curated", "glm-3.05": "curated", "glm-2.05": "curated"}


def make(quant="4.05", installed=("m305", "m405", "mglm205"), **cfg):
    """A Service with only what these paths use: the configured model running, three models installed."""
    svc = Service.__new__(Service)
    svc.cfg = dict(config.DEFAULTS, quant=quant, **cfg)
    svc.setup = types.SimpleNamespace(hw={"engine": {k: "1" for k in installed}}, blocking=False, running=None)
    svc.be = types.SimpleNamespace(state="running", tier="full")
    svc.bench = types.SimpleNamespace(running=False)
    svc.observe, svc.in_flight, svc.routing, svc.switching = False, 0, 0, None
    svc.events, svc.restarts = [], []
    svc.event = lambda kind, msg: svc.events.append((kind, msg))

    async def start(tier, reason):                 # Service.start: stops the other model's server, starts this one
        svc.restarts.append((svc.cfg["quant"], tier, reason))
        await asyncio.sleep(0.05)
        return True
    svc.start = start
    return svc


def body(model):
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()


class Case(unittest.TestCase):
    def setUp(self):
        self.cat = cat_with(**ALL)
        for target, value in ((catalog, "load"), (config, "save_config")):
            p = mock.patch.object(target, value, (lambda: self.cat) if value == "load" else self.saved)
            p.start()
            self.addCleanup(p.stop)
        self.saves = []

    def saved(self, cfg):
        self.saves.append(cfg["quant"])


class ListTests(Case):
    def test_the_installed_offered_models_the_configured_one_first(self):
        listed = Service.models_list(make("4.05"))
        self.assertEqual(listed["object"], "list")
        self.assertEqual([m["id"] for m in listed["data"]], [Q405, Q305, G205])     # GLM 3.05 isn't installed
        self.assertEqual(listed["data"][0], {"id": Q405, "object": "model", "created": DAY, "owned_by": "repertory",
                                             "max_model_len": 262144})
        self.assertEqual([m["id"] for m in Service.models_list(make("glm-2.05"))["data"]], [G205, Q305, Q405])
        self.assertEqual(Service.models_list(make("4.05", context=131072))["data"][2]["max_model_len"], 131072)

    def test_only_what_the_dashboard_offers_and_before_setup_has_looked_only_the_configured_one(self):
        self.cat = cat_with(**dict(ALL, **{"3.05": "failed"}))
        self.assertEqual([m["id"] for m in Service.models_list(make("4.05"))["data"]], [Q405, G205])
        self.assertEqual([m["id"] for m in Service.models_list(make("4.05", offer_uncurated=True))["data"]],
                         [Q405, Q305, G205])
        svc = make("4.05")
        svc.setup.hw = None
        self.assertEqual([m["id"] for m in Service.models_list(svc)["data"]], [Q405])
        self.cat = cat_with()                                                    # nothing curated yet
        self.assertEqual(Service.models_list(make("4.05"))["data"][0]["created"], 0)


class NameTests(Case):
    def resolve(self, name, quant="4.05", **kw):
        m, err = Service.resolve_model(make(quant, **kw), name)
        return (m["id"] if m else None), err

    def test_by_id_or_quant(self):
        self.assertEqual(self.resolve(Q305), (Q305, None))
        self.assertEqual(self.resolve(" 3.05 "), (Q305, None))
        self.assertEqual(self.resolve("glm-2.05"), (G205, None))

    def test_a_family_is_the_configured_model_if_it_is_of_it_else_its_largest_quant_installed(self):
        self.assertEqual(self.resolve("qwen3.8-flash-next", "3.05"), (Q305, None))
        self.assertEqual(self.resolve("qwen3.8-flash-next", "glm-2.05"), (Q405, None))
        self.assertEqual(self.resolve("glm-5.3-flash", "4.05"), (G205, None))

    def test_a_catalog_model_this_pc_cant_serve_is_an_error(self):
        m, err = self.resolve(G305)
        self.assertEqual(m, None)
        self.assertIn("isn't installed on this PC", err)
        self.cat = cat_with(**dict(ALL, **{"glm-2.05": "failed"}))
        self.assertIn("hasn't passed curation", self.resolve(G205)[1])
        self.assertIn("no glm-5.3-flash model is installed", self.resolve("glm-5.3-flash", installed=("m405",))[1])

    def test_any_other_name_is_for_the_configured_model(self):
        for name in ("repertory", "gpt-4o", "", None, 5, "qwen3.6-35b"):
            self.assertEqual(self.resolve(name), (None, None), name)


class RouteTests(Case):
    def route(self, svc, data):
        return asyncio.run(svc.route(data))

    def test_a_request_for_another_installed_model_switches_to_it(self):
        svc = make("4.05")
        self.assertEqual(self.route(svc, body(G205)), (True, "", 200, True))
        self.assertEqual(svc.cfg["quant"], "glm-2.05")
        self.assertEqual(self.saves, ["glm-2.05"])                               # the dashboard shows it too
        self.assertEqual(svc.restarts, [("glm-2.05", "full", "a request asked for GLM-5.3-Flash 2.05 bpw (lighter)")])
        self.assertEqual(svc.events[0][0], "switch")
        self.assertIsNone(svc.switching)
        self.assertEqual(svc.routing, 0)

    def test_the_model_it_has_or_a_name_it_doesnt_know_is_answered_as_it_is(self):
        svc = make("4.05")
        for data in (body(Q405), body("qwen3.8-flash-next"), body("repertory"), body(None), b"{}", b"not json", b""):
            self.assertEqual(self.route(svc, data), (True, "", 200, False), data)
        self.assertEqual((svc.restarts, svc.cfg["quant"]), ([], "4.05"))

    def test_a_model_it_cant_serve_is_refused_and_nothing_switches(self):
        svc = make("4.05")
        ok, msg, code, switched = self.route(svc, body(G305))
        self.assertEqual((ok, code, switched), (False, 404, False))
        self.assertIn("isn't installed", msg)
        self.assertEqual(svc.restarts, [])

    def test_switching_can_be_turned_off_and_waits_for_a_benchmark(self):
        svc = make("4.05", switch_on_request=False)
        self.assertEqual(self.route(svc, body(G205)), (True, "", 200, False))
        svc = make("4.05")
        svc.bench.running = True
        ok, msg, code, _ = self.route(svc, body(G205))
        self.assertEqual((ok, code), (False, 503))
        self.assertIn("benchmark", msg)
        self.assertEqual((svc.restarts, svc.cfg["quant"]), ([], "4.05"))

    def test_a_stopped_model_is_only_rechosen_the_start_is_left_to_the_request(self):
        svc = make("4.05")
        svc.be.state = "stopped"
        self.assertEqual(self.route(svc, body(Q305)), (True, "", 200, True))
        self.assertEqual((svc.cfg["quant"], svc.restarts), ("3.05", []))

    def test_the_switch_waits_for_running_requests_and_requests_meanwhile_wait_for_it(self):
        svc = make("4.05")

        async def go():
            svc.in_flight = 2                                      # one the model is answering, and this one
            switch = asyncio.create_task(svc.route(body(G205)))
            await asyncio.sleep(0.3)
            self.assertEqual((svc.switching, svc.restarts), ("glm-2.05", []))       # waiting for the running one
            svc.in_flight += 1
            later = asyncio.create_task(svc.route(body("repertory")))             # arrives during the switch
            await asyncio.sleep(0.3)
            self.assertFalse(later.done())
            svc.in_flight -= 1                                     # the running request ends
            self.assertEqual(await switch, (True, "", 200, True))
            self.assertEqual(await later, (True, "", 200, False))                   # then answered by GLM
            self.assertEqual(len(svc.restarts), 1)
        asyncio.run(go())

    def test_a_switch_gives_up_on_requests_that_keep_running(self):
        svc = make("4.05")
        svc.in_flight = 2
        with mock.patch.object(service_mod, "SWITCH_DRAIN_S", 0.3):
            ok, msg, code, switched = self.route(svc, body(G205))
        self.assertEqual((ok, code, switched, svc.cfg["quant"], svc.switching), (False, 503, False, "4.05", None))
        self.assertIn("still answering other requests", msg)


class Writer:
    def __init__(self):
        self.out = b""

    def write(self, b):
        self.out += b

    async def drain(self):
        pass

    def close(self):
        pass

    def get_extra_info(self, key):
        return ("127.0.0.1", 30000)


class HttpTests(Case):
    def get(self, svc, path):
        async def go():
            r = asyncio.StreamReader()
            r.feed_data(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:30000\r\n\r\n".encode())
            r.feed_eof()
            w = Writer()
            await svc.handle(r, w)
            return w.out
        raw = asyncio.run(go())
        head, _, data = raw.partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(data)

    def test_list_and_retrieve(self):
        svc = make("4.05")
        status, listed = self.get(svc, "/v1/models")
        self.assertEqual((status, [m["id"] for m in listed["data"]]), (200, [Q405, Q305, G205]))
        self.assertEqual(self.get(svc, f"/v1/models/{G205}"), (200, Service.model_object(svc, self.cat.find(G205))))
        self.assertEqual(self.get(svc, "/v1/models/glm-2.05")[1]["id"], G205)          # by quant too
        status, err = self.get(svc, "/v1/models/gpt-4o")
        self.assertEqual((status, err["error"]["code"], err["error"]["type"]),
                         (404, "model_not_found", "invalid_request_error"))
        self.assertIn("isn't installed", self.get(svc, f"/v1/models/{G305}")[1]["error"]["message"])


if __name__ == "__main__":
    unittest.main()

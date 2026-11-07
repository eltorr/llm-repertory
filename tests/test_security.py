"""The guards against other websites, other devices and hostile input: request parsing, origins, DNS rebinding, the
token, settings the HTTP API can't change, and figures read back from responses."""
import asyncio
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repertory import config  # noqa: E402
from repertory.httpio import HTTPError, backend_request, read_body, read_head  # noqa: E402
from repertory.service import Service, host_name, same_origin, token_count  # noqa: E402


def parse(raw, limit=1024):
    """read_head then read_body on raw request bytes."""
    async def go():
        r = asyncio.StreamReader()
        r.feed_data(raw)
        r.feed_eof()
        method, target, headers = await read_head(r)
        return method, target, headers, await read_body(r, None, headers, limit)
    return asyncio.run(go())


class ParserTests(unittest.TestCase):
    def test_plain_and_chunked_bodies(self):
        self.assertEqual(parse(b"POST /v1/x HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\nhi")[3], b"hi")
        raw = b"POST /v1/x HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n1;x=y\r\n!\r\n0\r\n\r\n"
        self.assertEqual(parse(raw)[3], b"hi!")

    def test_refused_heads(self):
        bad = [b"GET / HTTP/1.1\r\nX-A: 1\nTransfer-Encoding: chunked\r\n\r\n",   # bare LF: the backend would split
               b"GET / HTTP/1.1\r\nX-A: \x00\r\n\r\n",
               b"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n",
               b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n",
               b"POST / HTTP/1.1\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
               b"GET http://x/ HTTP/1.1\r\n\r\n",
               b"GET / HTTP/1.1\r\n folded: x\r\n\r\n"]
        for raw in bad:
            with self.assertRaises(HTTPError, msg=raw):
                parse(raw)

    def test_refused_bodies(self):
        for raw in (b"POST / HTTP/1.1\r\nContent-Length: 2000\r\n\r\n",              # over the limit, unread
                    b"POST / HTTP/1.1\r\nContent-Length: 0x3\r\n\r\nabc",
                    b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
                    b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n1_0\r\n",
                    b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n800\r\n" + b"x" * 2048 + b"\r\n0\r\n\r\n",
                    b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n"):
            with self.assertRaises(HTTPError, msg=raw):
                parse(raw)

    def test_token_and_key_never_reach_the_model_server(self):
        out = backend_request("POST", "/v1/x", [("Cookie", "repertory_token=t"), ("Authorization", "Bearer t"),
                                                ("X-Repertory-Key", "forged"), ("Content-Type", "a/b")], b"", 30001,
                              "real").decode()
        self.assertNotIn("repertory_token", out)
        self.assertNotIn("Bearer", out)
        self.assertNotIn("forged", out)
        self.assertIn("X-Repertory-Key: real", out)
        self.assertIn("Content-Type: a/b", out)


class OriginTests(unittest.TestCase):
    def test_programs_without_origin_pass(self):
        self.assertTrue(same_origin([("Host", "127.0.0.1:30000")]))

    def test_the_dashboard_itself_passes(self):
        self.assertTrue(same_origin([("Host", "127.0.0.1:30000"), ("Origin", "http://127.0.0.1:30000")]))
        self.assertTrue(same_origin([("host", "PC.Example:8080"), ("origin", "http://pc.example:8080")]))

    def test_other_sites_are_refused(self):
        for origin in ("https://evil.example", "http://127.0.0.1:8080", "null"):
            self.assertFalse(same_origin([("Host", "127.0.0.1:30000"), ("Origin", origin)]), origin)

    def test_host_name(self):
        for host, name in (("localhost:30000", "localhost"), ("[::1]:30000", "::1"), ("127.0.0.1", "127.0.0.1"),
                           ("Evil.Example:30000", "evil.example"), ("", ""), ("[bad", "")):
            self.assertEqual(host_name([("Host", host)]), name, host)


def service(**cfg):
    return types.SimpleNamespace(cfg={**config.DEFAULTS, "dashboard_token": "tok", **cfg}, observe=True,
                                 event=lambda *a: None, be=types.SimpleNamespace(state="stopped"))


class RequestTests(unittest.TestCase):
    def refusal(self, method, headers, front=False, **cfg):
        svc = service(network_port=8080, remote_host="pc.example", **cfg)
        svc.network_origin_ok = lambda h: Service.network_origin_ok(svc, h)
        return Service.refusal(svc, method, headers, front)

    def test_main_port_answers_local_names_only(self):
        self.assertIsNone(self.refusal("GET", [("Host", "127.0.0.1:30000")]))
        self.assertIsNone(self.refusal("POST", [("Host", "localhost:30000")]))
        self.assertEqual(self.refusal("GET", [("Host", "rebind.example:30000")])[0], 403)   # DNS rebinding

    def test_network_port_pages_must_come_from_this_pc(self):
        ok = [("Host", "pc.example:8080"), ("Origin", "http://pc.example:8080")]
        self.assertIsNone(self.refusal("POST", ok, front=True))
        ip = "[::ffff:127.0.0.1]:8080"     # any IP address passes: it can't be a rebound name
        self.assertIsNone(self.refusal("POST", [("Host", ip), ("Origin", "http://" + ip)], front=True))
        rebound = [("Host", "rebind.example:8080"), ("Origin", "http://rebind.example:8080")]
        self.assertEqual(self.refusal("POST", rebound, front=True)[0], 403)
        self.assertIsNone(self.refusal("POST", [("Host", "anything:8080")], front=True))   # programs: no Origin

    def test_cross_site_posts_are_refused(self):
        self.assertEqual(self.refusal("POST", [("Host", "127.0.0.1:30000"), ("Origin", "https://evil.example")])[0],
                         403)

    def test_token_from_cookie_or_bearer_and_from_the_link_only_on_the_page(self):
        svc = service()
        self.assertTrue(Service.token_ok(svc, [("Cookie", "a=b; repertory_token=tok")]))
        self.assertTrue(Service.token_ok(svc, [("Cookie", "qf_token=tok")]))      # signed in under QwenFlash
        self.assertTrue(Service.token_ok(svc, [("Authorization", "Bearer tok")]))
        self.assertTrue(Service.token_ok(svc, [], "/?token=tok"))
        self.assertFalse(Service.token_ok(svc, []))                 # an API route: ?token= isn't looked at
        self.assertFalse(Service.token_ok(svc, [("Cookie", "repertory_token=nope")], "/?token=nope"))
        self.assertFalse(Service.token_ok(service(dashboard_token=""), [("Cookie", "repertory_token=")]))

    def test_usage_figures_are_whole_numbers_in_range(self):
        self.assertEqual(token_count(12, 5), 12)
        self.assertEqual(token_count(None, 7), 7)
        for bad in ("x", 1e308, -1, True, 10**12, float("nan")):
            self.assertEqual(token_count(bad), 0, bad)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.svc.cfg["dashboard_token"] = ""

    def update(self, changes, local=True):
        return Service.update_config(self.svc, changes, local)

    def test_paths_programs_and_ports_only_in_config_json(self):
        for k, v in (("repo_dir", "/tmp/x'; id; '"), ("ui_browser", "C:/Windows/System32/cmd.exe"),
                     ("distro", "Other"), ("dashboard_token", "guess"), ("network_port", 8080)):
            status, _ = self.update({k: v})
            self.assertEqual(status, 400, k)
        self.assertEqual(self.svc.cfg, config.DEFAULTS)

    def test_other_devices_cannot_rename_the_pc(self):
        status, _ = self.update({"remote_host": "attacker.example"}, local=False)
        self.assertEqual(status, 403)
        self.assertEqual(self.svc.cfg["remote_host"], "")

    def test_a_bad_value_changes_nothing(self):
        status, _ = self.update({"watchlist": "game.exe", "remote_host": 'a" onmouseover="x'})
        self.assertEqual(status, 400)
        self.assertEqual(self.svc.cfg["watchlist"], [])

    def test_numbers_must_be_finite_and_in_range(self):
        for k, v in (("poll_s", "nan"), ("poll_s", "-1"), ("poll_s", "0"), ("lean_ram_gb", "inf"),
                     ("ram_lean_gb", "1e400"), ("context", 100), ("pressure_response", "reboot")):
            status, _ = self.update({k: v})
            self.assertEqual(status, 400, (k, v))
        self.assertEqual(self.svc.cfg, {**config.DEFAULTS, "dashboard_token": ""})

    def test_normal_settings_save(self):
        # the dashboard may resend a file-only setting unchanged
        status, r = self.update({"listen_port": "30000", "watchlist": "game.exe", "remote_host": "pc.example",
                                 "poll_s": "2", "pressure_response": "pause"})
        self.assertEqual(status, 200, r)
        self.assertEqual(self.svc.cfg["watchlist"], ["game.exe"])
        self.assertEqual(self.svc.cfg["remote_host"], "pc.example")
        self.assertEqual(self.svc.cfg["poll_s"], 2.0)

    def test_config_json_values_are_type_checked_on_load(self):
        config.CONFIG_PATH.write_text('{"lean_ram_gb": "a[$(id)]", "poll_s": NaN, "context": 65536}')
        cfg = config.load_config()
        self.assertEqual(cfg["lean_ram_gb"], config.DEFAULTS["lean_ram_gb"])
        self.assertTrue(math.isfinite(cfg["poll_s"]))
        self.assertEqual(cfg["context"], 65536)
        config.CONFIG_PATH.unlink()


class ModelServerTests(unittest.TestCase):
    """The model server's guard (engines/exl3_offload/server.py). Importing it loads no engine."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines" / "exl3_offload"))
        import server
        cls.LocalOnly = server.LocalOnly

    def call(self, headers, key="", local=True):
        """'app' when the request reaches the model, else the refusal's status."""
        reached, sent = [], []

        async def app(scope, receive, send):
            reached.append(scope)

        async def send(message):
            sent.append(message)
        scope = {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
        asyncio.run(self.LocalOnly(app, key, local)(scope, None, send))
        return "app" if reached else sent[0]["status"]

    def test_bound_to_localhost(self):
        self.assertEqual(self.call({"host": "127.0.0.1:30001"}), "app")
        self.assertEqual(self.call({"host": "[::1]:30001"}), "app")
        self.assertEqual(self.call({"host": "rebind.example:30001"}), 403)
        self.assertEqual(self.call({"host": "127.0.0.1:30001", "origin": "https://site.example"}), 403)

    def test_the_manager_key(self):
        self.assertEqual(self.call({"host": "127.0.0.1"}, key="k1"), 403)
        self.assertEqual(self.call({"host": "127.0.0.1", "x-repertory-key": "k2"}, key="k1"), 403)
        self.assertEqual(self.call({"host": "127.0.0.1", "x-repertory-key": "k1"}, key="k1"), "app")

    def test_bound_to_a_network_address(self):
        # --host with a network address: the operator chose to serve the network, so any host name, but still no
        # web pages
        self.assertEqual(self.call({"host": "lan-pc.example:30000"}, local=False), "app")
        self.assertEqual(self.call({"host": "lan-pc.example:30000", "origin": "https://site.example"}, local=False),
                         403)


if __name__ == "__main__":
    unittest.main()

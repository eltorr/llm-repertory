"""A setup step's result: the check after it, even with another check already running, and the CLI handing a step to
the running service (as the dashboard does) and reporting what the service found afterwards."""
import asyncio
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repertory import __main__ as cli  # noqa: E402
from repertory import setup as setup_mod  # noqa: E402


class CheckTests(unittest.TestCase):
    def setup_obj(self):
        svc = types.SimpleNamespace(host=types.SimpleNamespace(kind="linux"), cfg={})
        s = setup_mod.Setup(svc)
        s.calls = []

        def detect(host, cfg):
            s.calls.append(s.state)
            import time
            time.sleep(0.2)
            return {"state": s.state}
        s.statuses = lambda hw: {"build": hw["state"]}
        return s, detect

    def test_a_check_waits_for_one_running_and_reads_again(self):
        s, detect = self.setup_obj()
        s.state = "outdated"

        async def go():
            with mock.patch.object(setup_mod.hardware, "detect", detect):
                first = asyncio.create_task(s.check())       # e.g. the periodic one, started before the step ended
                await asyncio.sleep(0.05)
                s.state = "ok"                               # the step finished meanwhile
                await s.check()
                await first
        asyncio.run(go())
        self.assertEqual(s.calls, ["outdated", "ok"])        # read again, not the stale reading
        self.assertEqual(s.status["build"], "ok")


class FakeService(http.server.BaseHTTPRequestHandler):
    """The service's setup routes: the step runs for two polls, then its check for one, then reports ok."""
    polls, posted = 0, []

    def log_message(self, *a):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        assert self.headers.get("Authorization") == "Bearer tok"
        if "since=" not in self.path:
            return self.reply({"seq": 7, "lines": [], "running": None, "checking": False, "steps": []})
        FakeService.polls += 1
        n = FakeService.polls
        running = "build" if n <= 2 else None
        self.reply({"seq": 7 + n, "lines": [[7 + n, f"line {n}"]], "running": running, "checking": n == 3,
                    "steps": [{"id": "build", "status": "running" if n <= 3 else "ok"}]})

    def do_POST(self):
        FakeService.posted.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        self.reply({"ok": True, "message": "started"})


class CliTests(unittest.TestCase):
    def test_a_step_runs_in_the_service_and_waits_for_its_check(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/_switcher/api/"
            out = io.StringIO()
            with mock.patch.object(cli.time, "sleep", lambda s: None), redirect_stdout(out):
                rc = cli.setup_via_service({"dashboard_token": "tok"}, base, "build")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(rc, 0)
        self.assertEqual(FakeService.posted, [("/_switcher/api/setup/run", {"step": "build"})])
        self.assertEqual(out.getvalue().split(), ["line", "1", "line", "2", "line", "3", "line", "4"])


if __name__ == "__main__":
    unittest.main()

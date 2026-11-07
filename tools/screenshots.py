"""Dashboard screenshots for the docs, taken from the running service without changing anything.

    python tools/screenshots.py [--out docs/media] [--bench RUN]

Needs Playwright (pip install playwright; playwright install chromium) in the Python that runs it; it imports the
repertory package from this checkout for the dashboard token, which it never prints.

- Every request other than a GET is aborted, so no button can act on the service.
- The API answers are scrubbed before the page draws them: the dashboard token, this PC's names, the network port,
  addresses other than localhost and the user's folder become example values.
- Each image is saved only if the text on screen has none of those left.

Look at every image before committing it anyway."""
import argparse
import json
import os
import re
import secrets
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PORT_EXAMPLE = 8080
# installed Chromium browsers Playwright can drive when its own isn't downloaded (playwright install chromium)
BROWSERS = ["%ProgramFiles%/BraveSoftware/Brave-Browser/Application/brave.exe",
            "%ProgramFiles%/Google/Chrome/Application/chrome.exe", "%LOCALAPPDATA%/Google/Chrome/Application/chrome.exe",
            "%ProgramFiles(x86)%/Microsoft/Edge/Application/msedge.exe", "%ProgramFiles%/Microsoft/Edge/Application/msedge.exe",
            "/usr/bin/brave-browser", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/microsoft-edge"]


def launch(chromium, path=None):
    """Playwright's own Chromium if it is installed, else the first installed browser of BROWSERS (or path)."""
    if path is None:
        try:
            return chromium.launch()
        except Exception:  # noqa: BLE001 - its Chromium isn't downloaded: use an installed one
            path = next((os.path.expandvars(b) for b in BROWSERS if Path(os.path.expandvars(b)).is_file()), None)
            if path is None:
                raise
    return chromium.launch(executable_path=path)


def private_values(cfg):
    names = {cfg.get("remote_host") or "", os.environ.get("COMPUTERNAME", ""), socket.gethostname()}
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def make_scrub(cfg):
    token, port = cfg.get("dashboard_token") or "", cfg.get("network_port") or 0
    names, home, user = private_values(cfg), str(Path.home()), Path.home().name
    example_token = secrets.token_urlsafe(24)
    # whole LAN, tailnet (CGNAT) and unique-local addresses, not any dotted number: versions and Windows builds
    # (10.0.26200) look alike
    ipv4 = re.compile(r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])"
                      r"(?:\.\d{1,3}){2}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})(?![\d.])")
    ipv6 = re.compile(r"\b(?:f[cd][0-9a-f]{2}|fe80):[0-9a-f:]+\b", re.I)

    def text(s):
        if token:
            s = s.replace(token, example_token)
        for n in names:
            s = re.sub(re.escape(n), "gaming-pc", s, flags=re.I)
        s = s.replace(home, str(Path(home).parent / "you"))
        if len(user) >= 3:
            s = re.sub(rf"([\\/])(Users|home)([\\/]){re.escape(user)}\b", r"\1\2\3you", s, flags=re.I)
        if port:
            s = re.sub(rf":{port}\b", f":{PORT_EXAMPLE}", s).replace(f"port {port}", f"port {PORT_EXAMPLE}")
        return ipv6.sub("2001:db8::10", ipv4.sub("192.0.2.10", s))

    def value(v):
        if isinstance(v, str):
            return text(v)
        if isinstance(v, list):
            return [value(x) for x in v]
        if isinstance(v, dict):
            return {k: value(x) for k, x in v.items()}
        return PORT_EXAMPLE if port and v == port and not isinstance(v, bool) else v
    return value


def leaks(cfg, page_text):
    t = page_text.lower()
    words = [*private_values(cfg), cfg.get("dashboard_token") or "", f":{cfg.get('network_port') or 0}",
             str(Path.home()).lower()]
    return [w if len(w) < 20 else "<token>" for w in words if w and w.lower() in t]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "docs" / "media"))
    ap.add_argument("--live", default="http://127.0.0.1:30000")
    ap.add_argument("--bench", help="the benchmark run to show (a folder name under the state folder's bench/)")
    ap.add_argument("--browser", help="a Chromium-based browser to drive instead of Playwright's own")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from repertory import config
    cfg = config.load_config()
    scrub, blocked, errors, shots = make_scrub(cfg), [], [], {}

    def guard(route):
        req = route.request
        if req.method != "GET":
            blocked.append(f"{req.method} {req.url.split('?')[0]}")
            return route.abort()
        if "/_switcher/api/" not in req.url:
            return route.continue_()
        resp = route.fetch()
        try:
            body = json.dumps(scrub(json.loads(resp.text())))
        except ValueError:
            body = scrub(resp.text())
        return route.fulfill(response=resp, body=body)

    def keep(name, page, locator=None, clip=None):
        shown = (locator.inner_text() if locator is not None else page.locator("body").inner_text())
        found = leaks(cfg, shown)
        if found:
            print(f"NOT SAVED {name}: private values on screen: {found}")
            return
        shots[name] = locator.screenshot() if locator is not None else page.screenshot(clip=clip, full_page=True)

    def card(page, inner):
        return page.locator("section.card", has=page.locator(inner))

    with sync_playwright() as p:
        browser = launch(p.chromium, a.browser)
        for scheme in ("dark",):
            ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, color_scheme=scheme,
                                      device_scale_factor=1)
            ctx.add_cookies([{"name": "repertory_token", "value": cfg["dashboard_token"], "domain": "127.0.0.1",
                              "path": "/"}])
            pg = ctx.new_page()
            pg.route("**/*", guard)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(a.live + "/")
            pg.wait_for_selector(".tabs")
            pg.wait_for_timeout(5000)
            chart = card(pg, "#chart").bounding_box()
            keep("dashboard", pg, clip={"x": 0, "y": 0, "width": 1400, "height": chart["y"] + chart["height"] + 16})
            pg.click('.tabs button[data-tab="settings"]')
            pg.wait_for_timeout(1500)
            keep("dashboard-settings", pg, pg.locator("#p-settings"))
            # Setup: this machine's facts and the presets it recommends (the steps' state depends on the install)
            pg.click('.tabs button[data-tab="setup"]')
            pg.wait_for_selector("#recs :first-child", timeout=30000)
            pg.wait_for_timeout(1500)
            top, recs = pg.locator("#facts").bounding_box(), pg.locator("#recs").bounding_box()
            scroll = pg.evaluate("window.scrollY")        # boxes are relative to the window, a full-page clip isn't
            keep("dashboard-setup", pg, clip={"x": top["x"] - 8, "y": top["y"] + scroll - 8,
                                              "width": top["width"] + 16,
                                              "height": recs["y"] + recs["height"] - top["y"] + 16})
            pg.click('.tabs button[data-tab="bench"]')
            pg.wait_for_timeout(2000)
            run = a.bench or pg.eval_on_selector("#benchHistory", "s => s.options.length ? s.options[0].value : ''")
            if run:
                pg.select_option("#benchHistory", run)
                pg.wait_for_function("document.getElementById('benchTable').innerText.trim().length > 0",
                                     timeout=15000)       # a past run's results fill the table
                pg.wait_for_timeout(1500)
            keep("dashboard-bench", pg, pg.locator("#p-bench"))
            ctx.close()
        phone = browser.new_context(viewport={"width": 420, "height": 900}, color_scheme="dark", device_scale_factor=2)
        phone.add_cookies([{"name": "repertory_token", "value": cfg["dashboard_token"], "domain": "127.0.0.1",
                            "path": "/"}])
        pg = phone.new_page()
        pg.route("**/*", guard)
        pg.goto(a.live + "/")
        pg.wait_for_selector(".tabs")
        pg.wait_for_timeout(5000)
        keep("dashboard-phone", pg, clip={"x": 0, "y": 0, "width": 420, "height": 900})
        browser.close()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, png in shots.items():
        (out / f"{name}.png").write_bytes(png)
    print("page errors:", errors or "none")
    print("blocked:", blocked or "none")
    print("saved:", ", ".join(shots) or "nothing")


if __name__ == "__main__":
    main()

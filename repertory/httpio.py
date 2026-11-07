"""Minimal HTTP/1.1 plumbing on asyncio streams (stdlib only; one backend connection per request)."""
import asyncio
import json
import re

HOP = {"connection", "keep-alive", "proxy-connection", "host", "content-length", "transfer-encoding", "upgrade", "te",
       "expect"}
# never passed on to the model server: the dashboard's token, and the key only the manager may send
PRIVATE = {"cookie", "authorization", "proxy-authorization", "x-repertory-key"}
BACKEND_KEY_HEADER = "X-Repertory-Key"
REASONS = {200: "OK", 302: "Found", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
           405: "Method Not Allowed", 409: "Conflict", 413: "Content Too Large", 431: "Headers Too Large",
           501: "Not Implemented", 502: "Bad Gateway", 503: "Service Unavailable"}
# sent with every response: no framing (clickjacking), no type sniffing, no address in Referer
SECURITY_HEADERS = ("X-Frame-Options: DENY", "Content-Security-Policy: frame-ancestors 'none'",
                    "X-Content-Type-Options: nosniff", "Referrer-Policy: no-referrer")
MAX_HEAD = 64 * 1024
HEAD_TIMEOUT_S, BODY_TIMEOUT_S = 30, 300
TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


async def read_head(reader):
    """The request line and headers. Raises HTTPError for anything a strict parser would refuse: a bare CR, LF or NUL
    inside a line (the model server's parser would split there), malformed or repeated framing headers."""
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
    if len(head) > MAX_HEAD:
        raise HTTPError(431, "request headers too large")
    lines = head[:-4].split(b"\r\n")
    if any(b"\r" in ln or b"\n" in ln or b"\0" in ln for ln in lines):
        raise HTTPError(400, "bad characters in the request headers")
    parts = lines[0].decode("latin-1").split(" ")
    if len(parts) != 3 or not TOKEN.fullmatch(parts[0]) or not parts[1].startswith("/") \
            or not parts[2].startswith("HTTP/1."):
        raise HTTPError(400, "bad request line")
    headers = []
    for line in lines[1:]:
        k, sep, v = line.decode("latin-1").partition(":")
        if not sep or not TOKEN.fullmatch(k):
            raise HTTPError(400, "bad header line")
        headers.append((k, v.strip(" \t")))
    names = [k.lower() for k, _ in headers]
    if any(names.count(n) > 1 for n in ("host", "content-length", "transfer-encoding")) \
            or ("content-length" in names and "transfer-encoding" in names):
        raise HTTPError(400, "conflicting request headers")
    return parts[0], parts[1], headers


async def read_body(reader, writer, headers, limit):
    """The request body, at most `limit` bytes. Called only once the request has passed its checks."""
    h = {k.lower(): v for k, v in headers}
    te = h.get("transfer-encoding", "").lower()
    if te and te != "chunked":
        raise HTTPError(501, "unsupported transfer encoding")
    if "content-length" in h and not re.fullmatch(r"\d{1,15}", h["content-length"]):
        raise HTTPError(400, "bad Content-Length")
    size = int(h.get("content-length", "0"))
    if size > limit:
        raise HTTPError(413, f"request body over {limit // 1024 // 1024 or 1} MB")
    if h.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")      # curl & co. wait for this before sending large bodies
        await writer.drain()

    async def chunked():
        parts, total = [], 0
        while True:
            m = re.fullmatch(rb"([0-9A-Fa-f]{1,8})(;[^\r\n]*)?\r\n", await reader.readuntil(b"\r\n"))
            if not m:
                raise HTTPError(400, "bad chunk size")
            n = int(m.group(1), 16)
            total += n
            if total > limit:
                raise HTTPError(413, f"request body over {limit // 1024 // 1024 or 1} MB")
            if n == 0:
                while (await reader.readuntil(b"\r\n")) != b"\r\n":   # trailers, ignored
                    pass
                return b"".join(parts)
            parts.append(await reader.readexactly(n))
            if await reader.readexactly(2) != b"\r\n":
                raise HTTPError(400, "bad chunk")

    if te == "chunked":
        return await asyncio.wait_for(chunked(), BODY_TIMEOUT_S)
    return await asyncio.wait_for(reader.readexactly(size), BODY_TIMEOUT_S) if size else b""


def backend_request(method, target, headers, body, port, key=""):
    out = [f"{method} {target} HTTP/1.1"]
    out += [f"{k}: {v}" for k, v in headers if k.lower() not in HOP and k.lower() not in PRIVATE]
    out += [f"Host: 127.0.0.1:{port}", "Connection: close"]
    if key:
        out.append(f"{BACKEND_KEY_HEADER}: {key}")
    if body or method in ("POST", "PUT", "PATCH"):
        out.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body


async def respond(writer, status, body, ctype="application/json", extra=()):
    if not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode()
    head = [f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}", f"Content-Type: {ctype}",
            f"Content-Length: {len(body)}", "Cache-Control: no-store", "Connection: close", *SECURITY_HEADERS,
            *extra]
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
    try:
        await writer.drain()
    except ConnectionError:
        pass


def close(*writers):
    for w in writers:
        if w:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass


def oai_error(msg, kind="server_busy", code=503):
    return {"error": {"message": msg, "type": kind, "code": code}}


def http_get_json(url, timeout=2.0, headers=None):
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as r:
        return json.load(r)


def http_post_json(url, body, timeout=2.0, headers=None):
    """POSTs a JSON body; returns (status, the decoded JSON reply or None). A connection failure raises OSError."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, None


def http_ok(url, timeout=2.0, headers=None):
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


class StreamStats:
    """Watches a proxied completion response to time it: first token, end, and the usage block."""

    def __init__(self):
        self.t_first = None
        self.tail = b""
        self.head = b""
        self.headers_done = False
        self.events = 0
        self.responses = False
        self.status = None

    def feed(self, chunk, now):
        if not self.headers_done:
            self.head += chunk
            i = self.head.find(b"\r\n\r\n")
            if i < 0:
                return
            self.headers_done = True
            try:
                self.status = int(self.head.split(b" ", 2)[1])
            except (IndexError, ValueError):
                pass
            chunk = self.head[i + 4:]
        if self.t_first is None:
            if self.responses or b"event: response." in chunk:
                # Responses API: created / in_progress / item-added events come first; the first *.delta event
                # carries the first generated token
                self.responses = True
                if b'.delta"' in chunk:
                    self.t_first = now
            else:
                # chat SSE: the first event only announces the role; the second carries the first generated token
                self.events += chunk.count(b"data:")
                if self.events >= 2:
                    self.t_first = now
        self.tail = (self.tail + chunk)[-8192:]

    def usage(self):
        i = self.tail.rfind(b'"usage"')
        if i < 0:
            return None
        j = self.tail.find(b"{", i)
        depth = 0
        for k in range(j, len(self.tail)):
            depth += {ord("{"): 1, ord("}"): -1}.get(self.tail[k], 0)
            if depth == 0:
                try:
                    return json.loads(self.tail[j:k + 1])
                except ValueError:
                    return None
        return None


async def pipe(reader, writer, on_chunk=None):
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            break
        if on_chunk:
            on_chunk(chunk)
        writer.write(chunk)
        await writer.drain()

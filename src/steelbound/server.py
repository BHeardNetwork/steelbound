"""Local HTTP API for Steelbound.

Stdlib-only (`http.server`), bound to a loopback address and nothing else:
any non-loopback host is rejected before a socket is opened. The API is a
thin transport over the existing Core — it adds no new permissions. Chat,
recall, and ledger verification go through the same Orchestrator/Core paths
as the CLI, and tool runs go through ToolRegistry.run, so the deliberation
gate's rules are exactly unchanged (a WRITE_LOCAL or EXTERNAL tool still
needs explicit owner approval, passed as "approved" in the request body).

The server is intentionally single-threaded: Core holds one SQLite
connection, and sharing it across threads requires serialized access.
Requests are handled sequentially, which both satisfies that contract
and is the right shape for a local-only API.

The GUI shell lives here too: GET / (alias /console) serves
``console.html`` — a single self-contained page with no external
resources, so the console works fully offline. It is only a shell over
the API below: chat, recall, tools, and episodes all go through the
same Orchestrator/Core paths as the CLI, and the gate's rules are
exactly unchanged.

Endpoints (all JSON unless noted):
  GET  / or /console  -> the console page (text/html)
  GET  /status        -> {"version", "provider", "memory": {episodic,
                        semantic}, "scheduler": [{name, interval_seconds,
                        enabled, run_count, last_status}]}
  GET  /episodes      -> {"episodes": [{day, text, document_path}]}
  POST /episodes      {"day": "YYYY-MM-DD"} (optional, default today)
                      -> {"day", "text", "already_existed": bool}
  POST /chat          {"message": str} -> {"reply": str}
  GET  /recall?query=...&limit=N -> {"items": [{id, kind, content, source,
                                                confidence, tags}]}
  GET  /ledger/verify -> {"valid": bool, "count": int}
  GET  /tools         -> {"tools": [{name, description, risk, schema}]}
  POST /tools         {"name": str, "args": {...}, "approved": bool}
                      -> {"result": ...}  (alias: POST /tools/run)
Errors are JSON too: 400 malformed request, 403 gate denial, 404 unknown
tool or route, 405 wrong method. Bodies larger than 1 MiB are rejected.
"""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__
from .orchestrator import Orchestrator

MAX_BODY_BYTES = 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def create_server(core, host: str = "127.0.0.1", port: int = 0) -> HTTPServer:
    """Build (but do not start) the API server. Refuses non-loopback hosts."""
    if not _is_loopback(host):
        raise ValueError(
            f"Steelbound API binds to loopback only; refusing host {host!r}")
    orch = Orchestrator(core)
    handler_cls = _make_handler(core, orch)
    server = HTTPServer((host, port), handler_cls)
    return server


def serve(core, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve forever (blocking). Prints the bound address for the operator."""
    server = create_server(core, host, port)
    bound_host, bound_port = server.server_address[:2]
    core.ledger.append("api", "API_START", {"host": bound_host, "port": bound_port})
    print(f"Steelbound API listening on http://{bound_host}:{bound_port} (loopback only)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _make_handler(core, orch: Orchestrator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "SteelboundAPI/1.0"

        # -- plumbing -------------------------------------------------
        def log_message(self, *args):  # keep stdout clean; errors still surface
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": message})

        def _send_html(self) -> None:
            path = os.path.join(os.path.dirname(__file__), "console.html")
            try:
                with open(path, "rb") as fh:
                    body = fh.read()
            except FileNotFoundError:
                self._error(500, "console page is not installed"); return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = self.headers.get("Content-Length")
            if length is None:
                return None, "missing Content-Length"
            try:
                n = int(length)
            except ValueError:
                return None, "invalid Content-Length"
            if n < 0 or n > MAX_BODY_BYTES:
                return None, "request body too large"
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}, None
            try:
                data = json.loads(raw.decode())
            except (ValueError, UnicodeDecodeError):
                return None, "request body is not valid JSON"
            if not isinstance(data, dict):
                return None, "request body must be a JSON object"
            return data, None

        # -- routes ---------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/console"):
                self._send_html()
            elif path == "/status":
                self._handle_status()
            elif path == "/episodes":
                self._send(200, {"episodes": [
                    {"day": e.day, "text": e.text,
                     "document_path": e.document_path}
                    for e in (orch.episode_show(r.day)
                              for r in orch.episode_list())]})
            elif path == "/recall":
                self._handle_recall()
            elif path == "/ledger/verify":
                self._send(200, {"valid": core.ledger.verify(),
                                 "count": core.ledger.count()})
            elif path == "/tools":
                self._send(200, {"tools": [
                    {"name": t.name, "description": t.description,
                     "risk": t.risk.value, "schema": t.schema}
                    for t in core.tools.list()]})
            elif path in ("/chat", "/tools/run", "/episodes"):
                self._error(405, "use POST for this endpoint")
            else:
                self._error(404, f"unknown route: {path}")

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/chat":
                self._handle_chat()
            elif path == "/episodes":
                self._handle_episode_compose()
            elif path in ("/tools", "/tools/run"):
                self._handle_tool_run()
            elif path in ("/recall", "/ledger/verify", "/status", "/", "/console"):
                self._error(405, "use GET for this endpoint")
            else:
                self._error(404, f"unknown route: {path}")

        def _handle_status(self):
            jobs = orch.schedule_status()
            self._send(200, {
                "version": __version__,
                "provider": core.provider.name,
                "memory": {
                    "episodic": len(core.memory.all_active(kind="episodic")),
                    "semantic": len(core.memory.all_active(kind="semantic")),
                },
                "scheduler": [
                    {"name": j.name, "interval_seconds": j.interval_seconds,
                     "enabled": j.enabled, "run_count": j.run_count,
                     "last_status": j.last_status}
                    for j in jobs]})

        def _handle_episode_compose(self):
            data, err = self._read_json()
            if err:
                self._error(400, err); return
            day = data.get("day")
            if day is not None and not isinstance(day, str):
                self._error(400, "field 'day' must be a date string"); return
            try:
                episode = orch.episode_compose(day)
            except ValueError as exc:
                self._error(400, str(exc)); return
            self._send(200, {"day": episode.day, "text": episode.text,
                             "already_existed": episode.already_existed})

        def _handle_chat(self):
            data, err = self._read_json()
            if err:
                self._error(400, err); return
            message = data.get("message")
            if not isinstance(message, str) or not message.strip():
                self._error(400, "field 'message' must be a non-empty string"); return
            self._send(200, {"reply": orch.chat(message)})

        def _handle_recall(self):
            query = parse_qs(urlparse(self.path).query)
            text = (query.get("query") or query.get("q") or [""])[0]
            try:
                limit = int((query.get("limit") or ["8"])[0])
            except ValueError:
                self._error(400, "limit must be an integer"); return
            limit = max(1, min(limit, 100))
            items = core.memory.recall(text, limit=limit)
            self._send(200, {"items": [
                {"id": i.id, "kind": i.kind, "content": i.content,
                 "source": i.source, "confidence": i.confidence, "tags": i.tags}
                for i in items]})

        def _handle_tool_run(self):
            data, err = self._read_json()
            if err:
                self._error(400, err); return
            name = data.get("name")
            if not isinstance(name, str) or not name:
                self._error(400, "field 'name' must be a non-empty string"); return
            args = data.get("args") or {}
            if not isinstance(args, dict):
                self._error(400, "field 'args' must be a JSON object"); return
            approved = bool(data.get("approved", False))
            try:
                result = orch.run_tool(name, args, approved=approved)
            except KeyError:
                self._error(404, f"unknown tool: {name}"); return
            except PermissionError as exc:
                self._error(403, str(exc)); return
            self._send(200, {"result": result})

    return Handler

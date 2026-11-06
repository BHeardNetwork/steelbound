import json, os, sys, tempfile, threading, unittest, urllib.error, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound import __version__
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.server import create_server


def req(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get_content_type(), e.read()


def req_json(base, method, path, body=None):
    status, ctype, raw = req(base, method, path, body)
    return status, json.loads(raw.decode()) if raw else {}


class GuiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.server = create_server(self.core, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close()
        self.thread.join(timeout=5)
        self.core.close(); self.tmp.cleanup()

    def test_console_page_served_at_root_and_alias(self):
        for path in ("/", "/console"):
            status, ctype, raw = req(self.base, "GET", path)
            self.assertEqual(status, 200, path)
            self.assertEqual(ctype, "text/html", path)
            html = raw.decode()
            self.assertIn("Steelbound Console", html)
            # The console talks to the local API only: no external resource.
            self.assertNotIn("https://", html)
            self.assertNotIn('src="http', html)
            for endpoint in ("/status", "/chat", "/recall", "/tools", "/episodes"):
                self.assertIn(endpoint, html)

    def test_status_reports_real_vitals(self):
        self.orch.remember_fact("The owner's preferred language is English.")
        self.orch.remember_fact("The owner prefers tea.")
        status, body = req_json(self.base, "GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["version"], __version__)
        self.assertEqual(body["provider"], "offline")
        self.assertEqual(body["memory"]["semantic"], 2)
        self.assertGreaterEqual(body["memory"]["episodic"], 0)
        jobs = {j["name"]: j for j in body["scheduler"]}
        self.assertEqual(set(jobs), {"consolidation", "audit", "rsi_cycle"})
        self.assertTrue(all(j["enabled"] for j in jobs.values()))

    def test_episodes_list_compose_and_idempotency(self):
        status, body = req_json(self.base, "GET", "/episodes")
        self.assertEqual(status, 200)
        self.assertEqual(body["episodes"], [])
        status, made = req_json(self.base, "POST", "/episodes", {})
        self.assertEqual(status, 200)
        self.assertFalse(made["already_existed"])
        self.assertTrue(made["text"])
        status, again = req_json(self.base, "POST", "/episodes", {})
        self.assertEqual(status, 200)
        self.assertTrue(again["already_existed"])
        self.assertEqual(again["day"], made["day"])
        status, listed = req_json(self.base, "GET", "/episodes")
        self.assertEqual(len(listed["episodes"]), 1)
        self.assertEqual(listed["episodes"][0]["day"], made["day"])
        self.assertEqual(listed["episodes"][0]["text"], made["text"])
        self.assertTrue(self.core.ledger.verify())

    def test_episode_compose_bad_day(self):
        status, _ = req_json(self.base, "POST", "/episodes", {"day": "not-a-day"})
        self.assertEqual(status, 400)
        status, _ = req_json(self.base, "POST", "/episodes", {"day": 20261101})
        self.assertEqual(status, 400)

    def test_console_routes_reject_wrong_method(self):
        status, _ = req_json(self.base, "POST", "/status", {})
        self.assertEqual(status, 405)
        status, _ = req_json(self.base, "POST", "/", {})
        self.assertEqual(status, 405)
        status, _ = req_json(self.base, "GET", "/episodes/2026-11-01")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()

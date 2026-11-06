import json, os, sys, tempfile, threading, unittest, urllib.error, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.server import create_server


def req(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


class ApiTestCase(unittest.TestCase):
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

    def test_binds_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for bad in ("0.0.0.0", "192.168.1.10", "example.org"):
            with self.assertRaises(ValueError):
                create_server(self.core, bad, 0)

    def test_chat_round_trip_and_ledger(self):
        status, body = req(self.base, "POST", "/chat", {"message": "hello steelbound"})
        self.assertEqual(status, 200)
        self.assertIsInstance(body["reply"], str)
        self.assertTrue(body["reply"])
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("CHAT_TURN", actions)
        self.assertTrue(self.core.ledger.verify())

    def test_chat_requires_message(self):
        status, body = req(self.base, "POST", "/chat", {})
        self.assertEqual(status, 400)
        status, body = req(self.base, "POST", "/chat", {"message": "   "})
        self.assertEqual(status, 400)

    def test_recall_returns_stored_fact(self):
        self.orch.remember_fact("The owner's preferred language is English.")
        status, body = req(self.base, "GET", "/recall?query=language")
        self.assertEqual(status, 200)
        self.assertTrue(any("English" in i["content"] for i in body["items"]))
        self.assertTrue(all(i["kind"] in ("episodic", "semantic") for i in body["items"]))

    def test_recall_bad_limit(self):
        status, body = req(self.base, "GET", "/recall?query=x&limit=abc")
        self.assertEqual(status, 400)

    def test_ledger_verify(self):
        status, body = req(self.base, "GET", "/ledger/verify")
        self.assertEqual(status, 200)
        self.assertTrue(body["valid"])
        self.assertEqual(body["count"], self.core.ledger.count())
        self.assertGreater(body["count"], 0)

    def test_tools_list(self):
        status, body = req(self.base, "GET", "/tools")
        self.assertEqual(status, 200)
        by_name = {t["name"]: t for t in body["tools"]}
        self.assertIn("web_search", by_name)
        self.assertEqual(by_name["web_search"]["risk"], "external")
        self.assertIn("read_file", by_name)
        self.assertIn("write_file", by_name)

    def test_tool_run_gate_unchanged(self):
        # WRITE_LOCAL without approval: denied, exactly as the CLI gate does.
        status, body = req(self.base, "POST", "/tools",
                           {"name": "write_file",
                            "args": {"path": "note.txt", "content": "hi"}})
        self.assertEqual(status, 403)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "workspace", "note.txt")))
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("TOOL_DENIED", actions)
        # Same call with explicit approval: allowed.
        status, body = req(self.base, "POST", "/tools/run",
                           {"name": "write_file",
                            "args": {"path": "note.txt", "content": "hi"},
                            "approved": True})
        self.assertEqual(status, 200)
        self.assertIn("wrote", body["result"])
        status, body = req(self.base, "POST", "/tools",
                           {"name": "read_file", "args": {"path": "note.txt"}})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"], "hi")

    def test_tool_run_unknown_and_malformed(self):
        status, _ = req(self.base, "POST", "/tools", {"name": "nope", "args": {}})
        self.assertEqual(status, 404)
        status, _ = req(self.base, "POST", "/tools", {"args": {}})
        self.assertEqual(status, 400)
        # Raw malformed JSON body.
        r = urllib.request.Request(self.base + "/chat", data=b"{not json",
                                   method="POST",
                                   headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(r, timeout=10)
            self.fail("expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_unknown_route_and_wrong_method(self):
        status, _ = req(self.base, "GET", "/nope")
        self.assertEqual(status, 404)
        status, _ = req(self.base, "GET", "/chat")
        self.assertEqual(status, 405)
        status, _ = req(self.base, "POST", "/ledger/verify", {})
        self.assertEqual(status, 405)


if __name__ == "__main__":
    unittest.main()

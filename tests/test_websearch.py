import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.websearch import (SearchResult, WebSearch, HtmlEndpointFetcher,
                                  WEB_RESULT_CONFIDENCE)

PAGE = """
<html><body>
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Falpha&amp;rut=x">Alpha Title</a>
<a class="result__snippet" href="https://example.org/alpha">Alpha snippet text.</a>
<a rel="nofollow" class="result__a" href="https://example.org/beta">Beta Title</a>
<a class="result__snippet" href="https://example.org/beta">Beta snippet.</a>
<a class="nav" href="https://example.org/nav">Not a result</a>
</body></html>
"""

def stub_fetcher_factory(results, calls=None):
    def fetch(query, max_results):
        if calls is not None: calls.append((query, max_results))
        return list(results)
    return fetch

RESULTS = [
    SearchResult("https://example.org/alpha", "Alpha Title", "Alpha snippet."),
    SearchResult("https://example.org/beta", "Beta Title", "Beta snippet."),
]

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        self.core = Core(Settings(data_dir=self.tmp.name),
                         web_fetcher=stub_fetcher_factory(RESULTS, self.calls))
        self.orch = Orchestrator(self.core)
    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

class TestGate(Base):
    def test_unapproved_search_denied_before_network(self):
        with self.assertRaises(PermissionError):
            self.orch.web_search("alpha")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.core.memory.all_active(), [])
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("TOOL_DENIED", actions)

    def test_unapproved_tool_run_denied(self):
        with self.assertRaises(PermissionError):
            self.core.tools.run("web_search", {"query": "alpha"})
        self.assertEqual(self.calls, [])

    def test_approved_search_fetches_once(self):
        out = self.orch.web_search("alpha", approved=True)
        self.assertEqual(self.calls, [("alpha", 5)])
        self.assertEqual(len(out["results"]), 2)

class TestStorage(Base):
    def test_results_stored_as_episodic_with_url_provenance(self):
        out = self.orch.web_search("alpha beta", approved=True)
        self.assertEqual(len(out["stored_item_ids"]), 2)
        items = self.core.memory.all_active(kind="episodic")
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i.kind == "episodic" for i in items))
        self.assertEqual({i.source for i in items},
                         {"https://example.org/alpha", "https://example.org/beta"})
        self.assertTrue(all("Alpha Title" in i.content or "Beta Title" in i.content for i in items))
        self.assertTrue(all(i.confidence == WEB_RESULT_CONFIDENCE for i in items))
        self.assertTrue(all("web_search" in i.tags for i in items))
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("WEB_SEARCH_STORED", actions)
        self.assertTrue(self.core.ledger.verify())

    def test_stored_results_recallable_by_keyword(self):
        self.orch.web_search("alpha", approved=True)
        got = self.core.memory.recall("alpha")
        self.assertTrue(any("Alpha Title" in i.content and i.source == "https://example.org/alpha" for i in got))

    def test_web_results_not_semantic_facts(self):
        self.orch.web_search("alpha", approved=True)
        self.assertEqual(self.core.memory.all_active(kind="semantic"), [])

class TestValidation(unittest.TestCase):
    def test_empty_query_rejected(self):
        ws = WebSearch(fetcher=stub_fetcher_factory(RESULTS))
        with self.assertRaises(ValueError):
            ws.search("   ")

    def test_max_results_clamped(self):
        calls = []
        many = [SearchResult(f"https://example.org/{i}", f"Title {i}") for i in range(20)]
        ws = WebSearch(fetcher=stub_fetcher_factory(many, calls))
        self.assertEqual(len(ws.search("q", max_results=99)), 10)
        self.assertEqual(calls[0][1], 10)

    def test_non_web_urls_and_duplicates_filtered(self):
        mixed = [
            SearchResult("https://example.org/a", "A"),
            SearchResult("ftp://example.org/b", "B"),
            SearchResult("https://example.org/a", "A again"),
            SearchResult("https://example.org/c", ""),
        ]
        ws = WebSearch(fetcher=stub_fetcher_factory(mixed))
        got = ws.search("q")
        self.assertEqual([r.url for r in got], ["https://example.org/a"])

    def test_fetcher_error_propagates_and_stores_nothing(self):
        def boom(query, max_results): raise RuntimeError("network down")
        tmp = tempfile.TemporaryDirectory()
        try:
            core = Core(Settings(data_dir=tmp.name), web_fetcher=boom)
            orch = Orchestrator(core)
            with self.assertRaises(RuntimeError):
                orch.web_search("q", approved=True)
            self.assertEqual(core.memory.all_active(), [])
            core.close()
        finally:
            tmp.cleanup()

class TestHtmlParser(unittest.TestCase):
    def test_parse_fixture_page(self):
        got = HtmlEndpointFetcher.parse_html(PAGE)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].url, "https://example.org/alpha")
        self.assertEqual(got[0].title, "Alpha Title")
        self.assertEqual(got[0].snippet, "Alpha snippet text.")
        self.assertEqual(got[1].url, "https://example.org/beta")

    def test_endpoint_must_be_web_url(self):
        with self.assertRaises(ValueError):
            HtmlEndpointFetcher(endpoint="ftp://example.org/search")

if __name__ == "__main__":
    unittest.main()

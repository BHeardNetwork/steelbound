import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.gate import Risk
from steelbound.cli import main as cli_main

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.cog = self.core.cognition
    def tearDown(self):
        self.core.close(); self.tmp.cleanup()

class TestSelfQuery(Base):
    def test_six_answers_and_ledger(self):
        q = self.cog.self_query("run consolidation", goal="build queue: memory consolidation",
                                risk=Risk.READ, authority="owner", schedule="6h")
        self.assertFalse(q.blocked)
        self.assertEqual(q.who["name"], "Steelbound")
        self.assertEqual(q.who["version"], "1.0.0")
        self.assertEqual(q.who["provider"], "offline")
        self.assertEqual(q.who["generation"], 0)
        self.assertEqual(q.who["lineage"], "root")
        self.assertEqual(q.who["authority"], "owner")
        self.assertEqual(q.what, "run consolidation")
        self.assertEqual(q.why, "build queue: memory consolidation")
        self.assertIn(self.tmp.name, q.where)
        self.assertEqual(q.how, {"risk": "read", "approved": False})
        self.assertEqual(q.when["schedule"], "6h")
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("SELF_QUERY", actions)
        self.assertTrue(self.core.ledger.verify())

    def test_no_why_means_blocked_and_gap_ledgered(self):
        q = self.cog.self_query("do something vague", risk=Risk.EXTERNAL)
        self.assertTrue(q.blocked)
        self.assertIsNone(q.why)
        entry = [e for e in self.core.ledger.entries(100) if e.action == "SELF_QUERY"][-1]
        self.assertIsNone(entry.payload["why"])
        self.assertTrue(entry.payload["blocked"])

    def test_blank_goal_is_no_goal(self):
        q = self.cog.self_query("act", goal="   ")
        self.assertTrue(q.blocked)

    def test_empty_action_rejected(self):
        with self.assertRaises(ValueError):
            self.cog.self_query("  ")

    def test_unknown_risk_rejected(self):
        with self.assertRaises(ValueError):
            self.cog.self_query("act", goal="g", risk="teleport")

    def test_scheduled_self_query_detail(self):
        detail = self.cog.scheduled_self_query()
        self.assertFalse(detail["blocked"])
        self.assertEqual(detail["generation"], 0)
        self.assertEqual(detail["provider"], "offline")
        self.assertIn("SELF_QUERY",
                      [e.action for e in self.core.ledger.entries(100)])

    def test_orchestrator_path(self):
        q = self.orch.cognition_self_query("check ledger", goal="audit")
        self.assertFalse(q.blocked)

class TestDeliberation(Base):
    def _delib(self):
        return self.cog.deliberate(
            "search the web", ["search", "skip"],
            {"search": "owner asked for current prices", "skip": "no network needed"},
            "search", "owner request, gated", risk=Risk.EXTERNAL)

    def test_deliberation_stored_and_ledgered(self):
        d = self._delib()
        self.assertEqual(d.chosen, "search")
        self.assertEqual(d.options, ("search", "skip"))
        rows = self.cog.deliberations("search the web")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].evidence["skip"], "no network needed")
        self.assertIn("DELIBERATION",
                      [e.action for e in self.core.ledger.entries(100)])

    def test_deliberation_validation(self):
        with self.assertRaises(ValueError):
            self.cog.deliberate("a", [], {}, "x", "r")
        with self.assertRaises(ValueError):
            self.cog.deliberate("a", ["x"], {"x": "e"}, "y", "r")
        with self.assertRaises(ValueError):
            self.cog.deliberate("a", ["x"], {"x": "e"}, "x", "  ")
        with self.assertRaises(ValueError):
            self.cog.deliberate("a", ["x"], "not a dict", "x", "r")

    def test_approval_without_deliberation_refused_even_when_approved(self):
        d = self.cog.request_approval("search the web", Risk.EXTERNAL,
                                      approved=True)
        self.assertFalse(d.allowed)
        self.assertIn("deliberation", d.reason.lower())
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("DELIBERATION_REQUIRED", actions)
        # The gate was never consulted for the refused request.
        gate = [e for e in self.core.ledger.entries(100)
                if e.action == "GATE_DECISION"
                and e.payload.get("action") == "search the web"]
        self.assertEqual(gate, [])

    def test_approval_with_deliberation_follows_gate(self):
        d = self._delib()
        denied = self.cog.request_approval("search the web", Risk.EXTERNAL,
                                           deliberation_id=d.id, approved=False)
        self.assertFalse(denied.allowed)
        allowed = self.cog.request_approval("search the web", Risk.EXTERNAL,
                                            deliberation_id=d.id, approved=True)
        self.assertTrue(allowed.allowed)

    def test_mismatched_deliberation_refused(self):
        d = self._delib()
        # Wrong action.
        self.assertFalse(self.cog.request_approval(
            "delete everything", Risk.EXTERNAL, deliberation_id=d.id,
            approved=True).allowed)
        # Wrong risk.
        self.assertFalse(self.cog.request_approval(
            "search the web", Risk.DESTRUCTIVE, deliberation_id=d.id,
            approved=True).allowed)
        # Nonexistent record.
        self.assertFalse(self.cog.request_approval(
            "search the web", Risk.EXTERNAL, deliberation_id=9999,
            approved=True).allowed)

    def test_forbidden_still_denied_with_deliberation(self):
        d = self.cog.deliberate("spend money", ["spend", "skip"],
                                {"spend": "owner said so"}, "spend",
                                "owner instruction", risk=Risk.FORBIDDEN)
        self.assertFalse(self.cog.request_approval(
            "spend money", Risk.FORBIDDEN, deliberation_id=d.id,
            approved=True).allowed)

    def test_read_needs_no_deliberation(self):
        self.assertTrue(self.cog.request_approval(
            "read note", Risk.READ).allowed)

class TestUnknowns(Base):
    def test_record_list_resolve(self):
        u = self.cog.record_unknown("What is the owner's budget cap?",
                                    context="asked during planning")
        self.assertEqual(u.status, "open")
        self.assertEqual([x.id for x in self.cog.unknowns()], [u.id])
        self.assertEqual(len(self.cog.unknowns(status="open")), 1)
        fid = self.orch.remember_fact("The owner's budget cap is 500.",
                                      source="owner")
        resolved = self.cog.resolve_unknown(u.id, fid)
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(resolved.resolution_id, fid)
        self.assertEqual(self.cog.unknowns(status="open"), [])
        actions = [e.action for e in self.core.ledger.entries(100)]
        self.assertIn("UNKNOWN_RECORDED", actions)
        self.assertIn("UNKNOWN_RESOLVED", actions)

    def test_unknown_validation(self):
        with self.assertRaises(ValueError):
            self.cog.record_unknown("   ")
        u = self.cog.record_unknown("q?")
        with self.assertRaises(KeyError):
            self.cog.resolve_unknown(u.id, 99999)
        fid = self.orch.remember_fact("answer", source="owner")
        self.cog.resolve_unknown(u.id, fid)
        with self.assertRaises(ValueError):
            self.cog.resolve_unknown(u.id, fid)
        with self.assertRaises(KeyError):
            self.cog.resolve_unknown(99999, fid)
        with self.assertRaises(ValueError):
            self.cog.unknowns(status="bogus")

class TestMetacognition(Base):
    def test_report_on_healthy_core(self):
        m = self.cog.metacognition()
        self.assertEqual(m.failing_suites, ())
        self.assertTrue(all(v["passed"] == v["total"] for v in m.eval_scores.values()))
        self.assertIn("memory", m.eval_scores)
        self.assertIsNone(m.mean_semantic_confidence)
        self.assertIsNone(m.calibration_gap)
        self.assertEqual(m.open_unknowns, 0)
        self.assertIn("all eval suites pass", m.assessment)

    def test_calibration_counts_overconfident_corrections(self):
        fid = self.orch.remember_fact("Language is French.", source="owner")
        # remember_fact stores at default confidence 0.9 via fact().
        self.core.memory.correct(fid, "Language is English.", source="owner")
        m = self.cog.metacognition()
        self.assertEqual(m.corrected_semantic, 1)
        self.assertEqual(m.overconfident_corrections, 1)
        self.assertEqual(m.calibration_gap, 1.0)
        self.assertIn("calibration gap", m.assessment)

    def test_open_unknowns_counted(self):
        self.cog.record_unknown("unanswered?")
        m = self.cog.metacognition()
        self.assertEqual(m.open_unknowns, 1)
        self.assertIn("known unknown", m.assessment)

class TestComprehension(Base):
    def _promote_one(self):
        eid = self.core.memory.episode("The capital is Paris.", source="owner",
                                       confidence=0.95)
        result = self.core.consolidation.run()
        self.assertEqual(len(result.promoted), 1)
        return eid, result.promoted[0]

    def test_clean_check(self):
        self._promote_one()
        report = self.cog.comprehension_check()
        self.assertEqual(report.checked, 1)
        self.assertEqual(report.mismatches, ())

    def test_tampered_promotion_detected_and_ledgered(self):
        eid, sid = self._promote_one()
        self.core.conn.execute(
            "UPDATE memory_items SET content=? WHERE id=?",
            ("The capital is London.", sid))
        self.core.conn.commit()
        report = self.cog.comprehension_check()
        self.assertEqual(len(report.mismatches), 1)
        self.assertEqual(report.mismatches[0].episodic_id, eid)
        self.assertEqual(report.mismatches[0].semantic_id, sid)
        mism = [e for e in self.core.ledger.entries(100)
                if e.action == "COMPREHENSION_MISMATCH"]
        self.assertEqual(len(mism), 1)
        self.assertEqual(mism[0].payload["semantic_id"], sid)

    def test_missing_source_detected(self):
        eid, sid = self._promote_one()
        self.core.conn.execute("DELETE FROM memory_items WHERE id=?", (eid,))
        self.core.conn.commit()
        report = self.cog.comprehension_check()
        self.assertEqual(len(report.mismatches), 1)
        self.assertIn("missing", report.mismatches[0].detail)

    def test_corrected_promotion_skipped(self):
        _, sid = self._promote_one()
        self.core.memory.correct(sid, "The capital is Paris, France.",
                                 source="owner")
        report = self.cog.comprehension_check()
        self.assertEqual(report.checked, 0)
        self.assertEqual(report.mismatches, ())

    def test_non_promoted_semantic_not_checked(self):
        self.orch.remember_fact("A plain fact.", source="owner")
        report = self.cog.comprehension_check()
        self.assertEqual(report.checked, 0)

class TestCLI(Base):
    def test_cli_surface(self):
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "query", "run audit", "--goal", "audit"]),
                         0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "query", "vague action"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "deliberate", "search the web",
                                   "--options", '["search","skip"]',
                                   "--evidence", '{"search":"owner asked"}',
                                   "--chosen", "search",
                                   "--reason", "owner request",
                                   "--risk", "external"]), 0)
        # Approve without a deliberation id is refused, but the CLI still
        # reports the decision (exit 0: the refusal is the result).
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "approve", "search the web",
                                   "--risk", "external", "--approved"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "approve", "search the web",
                                   "--risk", "external",
                                   "--deliberation-id", "1", "--approved"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "unknown-record", "What is the cap?"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "unknowns"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "check"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "scheduled-check"]), 0)
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "metacognition"]), 0)
        # Bad risk class is a usage error, not a traceback.
        self.assertEqual(cli_main(["--data-dir", self.tmp.name, "cognition",
                                   "query", "act", "--risk", "teleport"]), 2)

if __name__ == "__main__":
    unittest.main()

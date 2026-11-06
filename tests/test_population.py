import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from steelbound.config import Settings, Core
from steelbound.orchestrator import Orchestrator
from steelbound.cli import main as cli_main
from steelbound.population import behaviour_descriptor, cell_key, _bucket

HELDOUT = [
    {"content": "heldout fact alpha", "source": "heldout", "confidence": 0.85},
    {"content": "heldout fact beta", "source": "heldout", "confidence": 0.87},
]


def _variation(auto):
    return {"kind": "consolidation_auto_threshold", "value": auto}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = Core(Settings(data_dir=self.tmp.name))
        self.orch = Orchestrator(self.core)
        self.pop = self.core.population

    def tearDown(self):
        self.core.close()
        self.tmp.cleanup()


class TestDescriptor(unittest.TestCase):
    def test_bucket_rounds_to_nearest_005(self):
        self.assertEqual(_bucket(0.86), 0.85)
        self.assertEqual(_bucket(0.84), 0.85)
        self.assertEqual(_bucket(0.88), 0.9)
        self.assertEqual(_bucket(0.9), 0.9)

    def test_cell_key_stable_and_genome_specific(self):
        genome_a = {"provider": "offline", "consolidation_auto_threshold": 0.86,
                    "consolidation_min_confidence": 0.5,
                    "auto_approve_local_writes": False}
        genome_b = dict(genome_a, consolidation_auto_threshold=0.84)
        genome_c = dict(genome_a, provider="ollama")
        self.assertEqual(cell_key(genome_a), cell_key(genome_b))
        self.assertNotEqual(cell_key(genome_a), cell_key(genome_c))
        desc = behaviour_descriptor(genome_a)
        self.assertEqual(desc["auto_threshold_bucket"], 0.85)
        self.assertEqual(desc["min_confidence_bucket"], 0.5)


class TestPopulationRun(Base):
    def test_run_archives_two_cells(self):
        # Parent default auto threshold is 0.9: neither held-out item
        # promotes for the parent. Child 0.86 promotes only beta (1);
        # child 0.80 promotes both (2) — different buckets, both elites.
        run = self.pop.run_population([_variation(0.86), _variation(0.80)],
                                      heldout_items=HELDOUT)
        self.assertEqual(run.archived_count(), 2)
        self.assertEqual(run.cells_occupied, 2)
        self.assertEqual(self.pop.diversity(), 2)
        elites = self.pop.elites()
        self.assertEqual(sorted(e.quality for e in elites), [1.0, 2.0])
        # The archive, not the living list, keeps the population:
        # both members were retired by the sweep.
        self.assertTrue(all(r.status == "retired"
                            for r in self.core.offspring.offspring()))
        self.assertTrue(self.core.ledger.verify())

    def test_same_cell_replacement_keeps_only_the_better(self):
        # 0.86 and 0.84 share the 0.85 bucket; the second promotes both
        # held-out items (2) against the first's one (1), so it replaces.
        run = self.pop.run_population([_variation(0.86), _variation(0.84)],
                                      heldout_items=HELDOUT)
        self.assertEqual(run.archived_count(), 2)
        self.assertEqual(self.pop.diversity(), 1)
        elite = self.pop.elites()[0]
        self.assertEqual(elite.quality, 2.0)
        self.assertEqual(elite.genome["consolidation_auto_threshold"], 0.84)
        self.assertTrue(run.results[1].replaced_elite)

    def test_tie_keeps_incumbent(self):
        # 0.84 and 0.83 share the 0.85 bucket and promote the same
        # held-out items: the second ties the incumbent and must not
        # displace it.
        run = self.pop.run_population([_variation(0.84), _variation(0.83)],
                                      heldout_items=HELDOUT)
        self.assertEqual(self.pop.diversity(), 1)
        self.assertEqual(self.pop.elites()[0].quality, 2.0)
        self.assertEqual(self.pop.elites()[0].genome["consolidation_auto_threshold"],
                         0.84)
        self.assertFalse(run.results[1].archived)
        self.assertFalse(run.results[1].replaced_elite)

    def test_regression_never_archived(self):
        # Child 0.95 promotes nothing while the parent (0.9) promotes
        # nothing either on 0.85/0.87 items... use a 0.92 item so the
        # parent beats the child: targeted regression, not archived.
        heldout = [{"content": "heldout fact gamma", "source": "heldout",
                    "confidence": 0.92}]
        run = self.pop.run_population([_variation(0.95)],
                                      heldout_items=heldout)
        self.assertEqual(run.results[0].verdict, "regression")
        self.assertFalse(run.results[0].archived)
        self.assertEqual(self.pop.diversity(), 0)

    def test_failed_variation_does_not_stop_run(self):
        run = self.pop.run_population(
            [{"kind": "consolidation_auto_threshold", "value": 2.0},
             _variation(0.80)],
            heldout_items=HELDOUT)
        self.assertEqual(run.results[0].verdict, "failed")
        self.assertEqual(run.results[1].verdict, "promotion_worthy")
        self.assertEqual(self.pop.diversity(), 1)

    def test_empty_variation_list_rejected(self):
        with self.assertRaises(ValueError):
            self.pop.run_population([])

    def test_run_is_ledgered(self):
        self.pop.run_population([_variation(0.80)], heldout_items=HELDOUT)
        actions = [e.action for e in self.core.ledger.entries(10000)]
        self.assertIn("POPULATION_ARCHIVED", actions)
        self.assertIn("POPULATION_RUN", actions)


class TestAdoptElite(Base):
    def _archived_elite(self):
        self.pop.run_population([_variation(0.80)], heldout_items=HELDOUT)
        return self.pop.elites()[0]

    def test_adopt_requires_approval(self):
        elite = self._archived_elite()
        with self.assertRaises(ValueError):
            self.pop.adopt_elite(elite.id, approved=False)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.9)

    def test_adopt_applies_and_persists_genome(self):
        elite = self._archived_elite()
        adopted = self.pop.adopt_elite(elite.id, approved=True)
        self.assertEqual(adopted.id, elite.id)
        self.assertEqual(self.core.consolidation.auto_threshold, 0.80)
        self.assertTrue(self.core.ledger.verify())
        # Adoption survives a restart via the adopted_genome row.
        self.core.close()
        reopened = Core(Settings(data_dir=self.tmp.name))
        try:
            self.assertEqual(reopened.consolidation.auto_threshold, 0.80)
            self.assertEqual(len(reopened.population.elites()), 1)
        finally:
            reopened.close()
        # Rebuild self.core handle for tearDown (it was closed above).
        self.core = Core(Settings(data_dir=self.tmp.name))

    def test_adopt_unknown_elite(self):
        with self.assertRaises(KeyError):
            self.pop.adopt_elite(999, approved=True)


class TestPopulationCLI(Base):
    def test_cli_run_list_adopt(self):
        rc = cli_main(["--data-dir", self.tmp.name, "population", "run",
                       "--variation",
                       '{"kind": "consolidation_auto_threshold", "value": 0.80}'])
        self.assertEqual(rc, 0)
        rc = cli_main(["--data-dir", self.tmp.name, "population", "list"])
        self.assertEqual(rc, 0)
        # Adopt without approval fails; with approval it applies.
        rc = cli_main(["--data-dir", self.tmp.name, "population", "adopt", "1"])
        self.assertEqual(rc, 1)
        rc = cli_main(["--data-dir", self.tmp.name, "population", "adopt", "1",
                       "--approved"])
        self.assertEqual(rc, 0)
        # CLI run without a variation is a usage error.
        rc = cli_main(["--data-dir", self.tmp.name, "population", "run"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

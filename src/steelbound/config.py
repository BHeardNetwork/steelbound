from __future__ import annotations
import os, sqlite3
from dataclasses import dataclass
from .ledger import Ledger
from .memory import Memory
from .consolidation import Consolidator
from .scheduler import Scheduler
from .gate import Gate
from .tools import Tool, ToolRegistry, make_file_tools
from .gate import Risk
from .websearch import WebSearch
from .rsi import RSIEngine
from .offspring import OffspringEngine, load_adoption
from .cognition import CognitionEngine
from .episode import EpisodeEngine
from .prediction import PredictionEngine
from .population import PopulationEngine
from .synthesis import SynthesisEngine, load_synthesized_tools
from .ecosystem import EcosystemEngine
from .toolpacks import ToolpackManager, load_toolpacks
from .providers import make_provider, Provider

@dataclass
class Settings:
    data_dir: str = ".steelbound"
    provider: str = "offline"
    provider_kwargs: dict | None = None
    auto_approve_local_writes: bool = False
    @property
    def db_path(self) -> str: return os.path.join(self.data_dir, "steelbound.db")
    @property
    def workspace(self) -> str: return os.path.join(self.data_dir, "workspace")

class Core:
    """One object wiring every component. This is the whole runtime."""
    def __init__(self, settings: Settings | None = None, provider: Provider | None = None,
                 web_fetcher=None):
        self.settings = settings or Settings()
        os.makedirs(self.settings.workspace, exist_ok=True)
        # check_same_thread=False so the local API server (which runs in its
        # own thread) can use this Core. Callers that share a Core across
        # threads must serialize access; the API server is single-threaded
        # and the CLI is single-threaded, so both satisfy that contract.
        self.conn = sqlite3.connect(self.settings.db_path, check_same_thread=False)
        self.ledger = Ledger(self.conn)
        self.memory = Memory(self.conn, self.ledger)
        self.consolidation = Consolidator(self.memory, self.ledger)
        # RSI R2: an auto-adopted threshold change persists in the
        # database; apply it on startup so adoption survives a restart.
        self.consolidation.load_persisted()
        self.scheduler = Scheduler(self.ledger, self.conn, self.consolidation)
        self.rsi = RSIEngine(self.ledger, self.conn, self.memory,
                             self.consolidation, self.scheduler)
        # RSI R3: bind the scheduled-recursion job body. The scheduler
        # is built before the engine, so this binding happens here,
        # on every Core construction (including reopen after restart).
        self.scheduler.bind_rsi_cycle(self.rsi.scheduled_cycle)
        self.gate = Gate(self.ledger, self.settings.auto_approve_local_writes)
        self.tools = ToolRegistry(self.gate, self.ledger)
        for t in make_file_tools(self.settings.workspace): self.tools.register(t)
        # Web search is EXTERNAL: the gate denies it without explicit owner
        # approval, before any network call. The fetcher is pluggable.
        self.websearch = WebSearch(fetcher=web_fetcher)
        self.tools.register(Tool(
            "web_search", "Search the web. Requires explicit owner approval.",
            Risk.EXTERNAL, self.websearch.search_as_dicts,
            {"query": "str", "max_results": "int"}))
        # Synthesis S3: kept synthesized tools are re-registered from
        # the database, so acceptance survives a restart.
        load_synthesized_tools(self)
        self.provider = provider or make_provider(self.settings.provider, **(self.settings.provider_kwargs or {}))
        # Offspring promotion persists the adopted provider/gate part in
        # the database; a reopened core re-applies it before its first
        # ledger entry, so promotion survives a restart.
        adopted = load_adoption(self.conn)
        if adopted and provider is None:
            self.provider = make_provider(adopted["provider"],
                                          **adopted["provider_kwargs"])
            self.gate.auto_approve_local_writes = adopted["auto_approve_local_writes"]
        self.offspring = OffspringEngine(self.ledger, self.conn, core=self)
        # Synthesis S3: quality-diversity population archive over
        # offspring evaluations, and sandboxed program synthesis.
        # Built after offspring (population spawns through it) and
        # after the tool registry (synthesis registers kept tools).
        self.population = PopulationEngine(self.ledger, self.conn, core=self)
        self.synthesis = SynthesisEngine(self.ledger, self.conn, self.tools)
        # Ecosystem registry: apps, offspring, toolpacks, and services
        # around this core, each with permissions a subset of this
        # core's ceiling. Built after the gate and tool registry exist,
        # since registration and retirement are gated through them.
        self.ecosystem = EcosystemEngine(self.ledger, self.conn, core=self)
        # Toolpack SDK: operator-installed packs persist their load
        # records in the database and reload on startup, registering
        # as ecosystem members (kind "toolpack") when they do. Built
        # after the ecosystem registry so that hook finds it in place.
        self.toolpacks = ToolpackManager(self.ledger, self.conn,
                                         tools=self.tools, gate=self.gate,
                                         core=self)
        load_toolpacks(self)
        # Cognition: self-query, metacognition, deliberation records.
        # Built last so its identity can read the provider, data dir,
        # and any OFFSPRING_GENESIS entry already in the ledger.
        self.cognition = CognitionEngine(
            self.ledger, self.conn, memory=self.memory, gate=self.gate,
            provider_name=self.provider.name,
            data_dir=self.settings.data_dir)
        # Predictive inference (Synthesis S2): predict outcomes before
        # they happen, measure surprise, and feed surprising outcomes
        # into cognition's known unknowns. Built after cognition so
        # promote_to_unknowns has an engine to write into.
        self.prediction = PredictionEngine(
            self.ledger, self.conn, cognition=self.cognition)
        # Daily episode: narrative of the core's own day, from its
        # records. Built last so its tomorrow-plan can read the
        # scheduler, consolidator, RSI engine, and cognition engine.
        self.episode = EpisodeEngine(
            self.ledger, self.conn, self.settings.data_dir,
            memory=self.memory, scheduler=self.scheduler,
            consolidation=self.consolidation, rsi=self.rsi,
            cognition=self.cognition)
        self.ledger.append("core", "CORE_INIT", {"provider": self.provider.name})
    def close(self): self.conn.close()

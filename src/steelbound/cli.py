from __future__ import annotations
import argparse, json, sys
from .config import Settings, Core
from .orchestrator import Orchestrator
from . import evals

def build_core(args) -> Core:
    return Core(Settings(data_dir=args.data_dir, provider=args.provider,
                         auto_approve_local_writes=args.auto_local))

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="steelbound", description="Steelbound — local-first AI core. Everything significant in the ledger.")
    p.add_argument("--data-dir", default=".steelbound"); p.add_argument("--provider", default="offline")
    p.add_argument("--auto-local", action="store_true", help="auto-approve local sandbox writes")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo"); sub.add_parser("evals")
    c = sub.add_parser("chat"); c.add_argument("message")
    r = sub.add_parser("remember"); r.add_argument("fact")
    q = sub.add_parser("recall"); q.add_argument("query")
    led = sub.add_parser("ledger"); led.add_argument("action", choices=["verify", "list", "count"])
    t = sub.add_parser("tool"); t.add_argument("name"); t.add_argument("--args", default="{}"); t.add_argument("--approved", action="store_true")
    s = sub.add_parser("search", help="gated web search; results stored as episodic memory with URL provenance")
    s.add_argument("query"); s.add_argument("--max-results", type=int, default=5); s.add_argument("--approved", action="store_true")
    cn = sub.add_parser("consolidate", help="episodic -> semantic consolidation")
    cn.add_argument("action", nargs="?", default="run", choices=["run", "pending", "approve", "reject"])
    cn.add_argument("proposal_id", nargs="?", type=int)
    ex = sub.add_parser("export", help="export memory + ledger as one JSON bundle")
    ex.add_argument("--out", default=None, help="write bundle to this file (default: stdout)")
    im = sub.add_parser("import", help="import a JSON bundle; its chain is re-verified first")
    im.add_argument("path")
    sv = sub.add_parser("serve", help="local HTTP API, loopback only (127.0.0.1)")
    sv.add_argument("--host", default="127.0.0.1", help="loopback host only; anything else is refused")
    sv.add_argument("--port", type=int, default=8765)
    bk = sub.add_parser("backup", help="one-command snapshot of the data dir into a zip file")
    bk.add_argument("--out", default=None, help="backup file (default: steelbound-backup-YYYYMMDD-HHMMSS.zip in the current directory)")
    rs = sub.add_parser("restore", help="restore a backup zip into a data dir; verifies it first, refuses a non-empty target unless --force")
    rs.add_argument("path")
    rs.add_argument("--into", default=None, help="target data dir (default: --data-dir)")
    rs.add_argument("--force", action="store_true", help="replace an existing database/workspace in the target")
    sc = sub.add_parser("schedule", help="in-core scheduler: recurring consolidation (6h) + audit self-check (1h) + RSI scheduled recursion (6h)")
    sc.add_argument("action", nargs="?", default="status",
                    choices=["status", "run", "enable", "disable", "interval"])
    sc.add_argument("name", nargs="?", help="job name for run/enable/disable/interval")
    sc.add_argument("value", nargs="?", help="interval seconds for 'interval'")
    rs_ = sub.add_parser("rsi", help="bounded self-improvement R0-R3: observe, propose, sandbox-evaluate; all-green config patches may auto-adopt, code stays owner-approved, scheduled recursion via the rsi_cycle scheduler job")
    rs_.add_argument("action", nargs="?", default="pending",
                     choices=["observe", "propose", "pending", "evaluate", "approve", "reject", "cycle", "submit", "auto-adopt", "rollback", "rollback-check", "history"])
    rs_.add_argument("proposal_id", nargs="?", type=int)
    rs_.add_argument("--kind", default="code")
    rs_.add_argument("--target", default="")
    rs_.add_argument("--rationale", default="")
    rs_.add_argument("--patch", default=None, help="patch JSON for 'submit'")
    of = sub.add_parser("offspring", help="offspring cores: spawn a child with one variation, head-to-head eval vs the parent, owner-approved promote or retire")
    of.add_argument("action", nargs="?", default="list",
                    choices=["list", "spawn", "evaluate", "promote", "retire"])
    of.add_argument("offspring_id", nargs="?", type=int)
    of.add_argument("--variation", default=None, help="variation JSON for 'spawn', e.g. {\"kind\": \"consolidation_auto_threshold\", \"value\": 0.85}")
    of.add_argument("--sandbox-dir", default=None, help="sandbox data dir for 'spawn' (default: <data-dir>/offspring/gen-N)")
    of.add_argument("--seed-limit", type=int, default=None, help="max semantic items to seed the child with")
    of.add_argument("--seed-min-confidence", type=float, default=0.0, help="minimum confidence of seeded semantic items")
    of.add_argument("--heldout", default=None, help="JSON file of held-out episodic items for 'evaluate'")
    of.add_argument("--approved", action="store_true", help="explicit owner approval for 'promote'")
    of.add_argument("--reason", default="", help="reason for 'retire'")
    cg = sub.add_parser("cognition", help="cognition core: six-question self-query, metacognition, deliberation records")
    cg.add_argument("action", nargs="?", default="query",
                    choices=["query", "deliberate", "approve", "unknown-record", "unknowns",
                             "unknown-resolve", "metacognition", "check", "scheduled-check"])
    cg.add_argument("subject", nargs="?", help="action text for query/deliberate/approve, question for unknown-record, unknown id for unknown-resolve")
    cg.add_argument("--goal", default=None, help="why: goal, queue item, or owner instruction for 'query' (required for the query to be unblocked)")
    cg.add_argument("--where", default=None, help="where the action acts, for 'query'")
    cg.add_argument("--risk", default=None, help="risk class: read, write_local, external, destructive, forbidden")
    cg.add_argument("--approved", action="store_true", help="explicit owner approval for 'query'/'approve'")
    cg.add_argument("--authority", default="owner", help="on whose authority, for 'query'")
    cg.add_argument("--schedule", default=None, help="schedule name, for 'query'")
    cg.add_argument("--options", default=None, help="JSON list of options for 'deliberate'")
    cg.add_argument("--evidence", default=None, help="JSON dict mapping option to evidence for 'deliberate'")
    cg.add_argument("--chosen", default=None, help="chosen option for 'deliberate'")
    cg.add_argument("--reason", default="", help="deciding reason for 'deliberate'")
    cg.add_argument("--deliberation-id", type=int, default=None, help="deliberation record id for 'approve'")
    cg.add_argument("--resolution-id", type=int, default=None, help="memory item id grounding 'unknown-resolve'")
    cg.add_argument("--context", default="", help="context for 'unknown-record'")
    ep = sub.add_parser("episode", help="daily episode: first-person narrative of the core's day, from its own records, stored as a dated document + episodic memory")
    ep.add_argument("action", nargs="?", default="compose",
                    choices=["compose", "show", "list"])
    ep.add_argument("--day", default=None, help="day as YYYY-MM-DD (default: today; compose is idempotent per day)")
    pr = sub.add_parser("predict", help="predictive inference (Synthesis S2): predict outcomes before they happen, measure surprise in bits, queue the most surprising outcomes to learn, and promote surprises to cognition known unknowns")
    pr.add_argument("action", nargs="?", default="summary",
                    choices=["predict", "observe", "summary", "queue", "learn",
                             "promote", "review"])
    pr.add_argument("subject", nargs="?", help="stream for predict/summary, stream for observe (with --actual), outcome id for learn")
    pr.add_argument("--actual", default=None, help="observed outcome for 'observe'")
    pr.add_argument("--context", default="", help="context for predict/observe")
    pr.add_argument("--prediction-id", type=int, default=None, help="prediction id resolving for 'observe'")
    pr.add_argument("--limit", type=int, default=10, help="queue/promote limit")
    pr.add_argument("--threshold", type=float, default=None, help="surprise threshold in bits for 'promote'")
    pp = sub.add_parser("population", help="quality-diversity population evolution (Synthesis S3): spawn, evaluate, and archive one member per variation, one elite per behaviour cell; adoption stays owner-approved")
    pp.add_argument("action", nargs="?", default="list",
                    choices=["list", "run", "adopt"])
    pp.add_argument("elite_id", nargs="?", type=int, help="archived elite id for 'adopt'")
    pp.add_argument("--variation", action="append", default=None, help="variation JSON for 'run' (repeatable), e.g. {\"kind\": \"consolidation_auto_threshold\", \"value\": 0.85}")
    pp.add_argument("--heldout", default=None, help="JSON file of held-out episodic items for 'run'")
    pp.add_argument("--approved", action="store_true", help="explicit owner approval for 'adopt'")
    sy = sub.add_parser("synthesize", help="program synthesis (Synthesis S3): propose a pure tool as code; kept only when every test case passes in the sandbox")
    sy.add_argument("action", nargs="?", default="list",
                    choices=["propose", "list", "remove"])
    sy.add_argument("name", nargs="?", help="tool name for propose/remove")
    sy.add_argument("--description", default="", help="tool description for 'propose'")
    sy.add_argument("--code", default=None, help="tool source for 'propose' (a single 'run' function)")
    sy.add_argument("--code-file", default=None, help="file holding the tool source for 'propose'")
    sy.add_argument("--cases", default=None, help="test cases JSON for 'propose': [{\"args\": {...}, \"expected\": ...}]")
    sy.add_argument("--cases-file", default=None, help="file holding the test cases JSON for 'propose'")
    sy.add_argument("--timeout", type=float, default=10.0, help="sandbox timeout in seconds for 'propose'")
    ec = sub.add_parser("ecosystem", help="ecosystem registry: apps, offspring, toolpacks, and services around this core, each with permissions a subset of the core's ceiling; registration is a gated local write, retirement is destructive")
    ec.add_argument("action", nargs="?", default="list",
                    choices=["register", "list", "inspect", "retire"])
    ec.add_argument("subject", nargs="?", help="manifest JSON file for 'register', member name for 'inspect'/'retire'")
    ec.add_argument("--approved", action="store_true", help="explicit owner approval for 'register'/'retire'")
    ec.add_argument("--reason", default="", help="reason for 'retire'")
    ec.add_argument("--status", default=None, help="status filter for 'list': active or retired")
    tp = sub.add_parser("toolpack", help="toolpack SDK: load, unload, or list external tool packs (statically validated; declared risk clamped by the loader; loading is a gated local write)")
    tp.add_argument("action", nargs="?", default="list",
                    choices=["load", "unload", "list"])
    tp.add_argument("path", nargs="?", help="pack directory for 'load', pack name for 'unload'")
    tp.add_argument("--approved", action="store_true", help="explicit owner approval for 'load'/'unload'")
    args = p.parse_args(argv)
    if args.cmd == "restore":
        # Deliberately before build_core: building a Core in the target
        # would create a database there before the non-empty check runs.
        from .backup import restore_backup
        target = args.into or args.data_dir
        try:
            summary = restore_backup(args.path, target, force=args.force)
        except (ValueError, OSError) as exc:
            print(f"restore rejected: {exc}", file=sys.stderr); return 1
        print(f"restored to {summary['data_dir']}: "
              f"{summary['ledger_entries']} ledger entries, "
              f"{summary['files_restored']} workspace files; chain valid: True")
        return 0
    if args.cmd == "evals":
        results = evals.run_all()
        for name, passed, total, failures in results: print(f"{name}: {passed}/{total}" + (f" FAIL {failures}" if failures else " PASS"))
        return 0 if evals.all_passed(results) else 1
    core = build_core(args); orch = Orchestrator(core)
    try:
        if args.cmd == "demo":
            orch.remember_fact("The project is Steelbound, a local-first AI core.", source="demo")
            print(orch.chat("What is the project, based on memory?"))
            print(orch.chat("remember: demo run completed"))
            print("ledger entries:", core.ledger.count(), "| chain valid:", core.ledger.verify())
        elif args.cmd == "chat": print(orch.chat(args.message))
        elif args.cmd == "remember": print("stored fact id:", orch.remember_fact(args.fact))
        elif args.cmd == "recall":
            for i in core.memory.recall(args.query): print(f"[{i.kind} via {i.source}] {i.content}")
        elif args.cmd == "ledger":
            if args.action == "verify": print("chain valid:", core.ledger.verify())
            elif args.action == "count": print(core.ledger.count())
            else:
                for e in core.ledger.entries(50): print(e.id, e.actor, e.action, e.entry_hash[:12])
        elif args.cmd == "tool":
            print(orch.run_tool(args.name, json.loads(args.args), approved=args.approved))
        elif args.cmd == "search":
            try:
                out = orch.web_search(args.query, max_results=args.max_results, approved=args.approved)
            except PermissionError:
                print("web search requires explicit owner approval: rerun with --approved"); return 1
            for r in out["results"]:
                print(f"- {r['title']}\n  {r['url']}" + (f"\n  {r['snippet']}" if r["snippet"] else ""))
            print("stored as episodic memory ids:", out["stored_item_ids"])
        elif args.cmd == "consolidate":
            if args.action == "run":
                r = orch.consolidate()
                print(f"promoted: {list(r.promoted)} | queued: {list(r.queued)} | skipped: {list(r.skipped)}")
            elif args.action == "pending":
                for pr in core.consolidation.pending():
                    print(f"proposal {pr.id}: episodic {pr.episodic_id} (conf {pr.confidence}) {pr.content!r}")
            elif args.action in ("approve", "reject"):
                if args.proposal_id is None:
                    print(f"consolidate {args.action} requires a proposal id"); return 2
                if args.action == "approve":
                    print("promoted to semantic id:", core.consolidation.approve(args.proposal_id))
                else:
                    core.consolidation.reject(args.proposal_id); print("rejected proposal", args.proposal_id)
        elif args.cmd == "schedule":
            sched = core.scheduler
            if args.action == "status":
                for j in sched.jobs():
                    last = "never" if j.last_run is None else f"{j.last_run:.0f}"
                    print(f"{j.name}: interval={j.interval_seconds:g}s "
                          f"enabled={int(j.enabled)} last_run={last} "
                          f"runs={j.run_count} last_status={j.last_status}")
            elif args.action == "run":
                try:
                    runs = [sched.run_job(args.name)] if args.name else sched.run_due()
                except (KeyError, ValueError) as exc:
                    print(f"schedule run failed: {exc}", file=sys.stderr); return 1
                if not runs:
                    print("no scheduler jobs due")
                for r in runs:
                    print(f"{r.job}: {r.status} {r.detail}")
            elif args.action in ("enable", "disable"):
                if args.name is None:
                    print(f"schedule {args.action} requires a job name"); return 2
                try:
                    job = sched.enable(args.name) if args.action == "enable" else sched.disable(args.name)
                except KeyError as exc:
                    print(f"schedule failed: {exc}", file=sys.stderr); return 1
                print(f"{job.name}: enabled={int(job.enabled)}")
            elif args.action == "interval":
                if args.name is None or args.value is None:
                    print("schedule interval requires a job name and seconds"); return 2
                try:
                    job = sched.set_interval(args.name, float(args.value))
                except (KeyError, ValueError) as exc:
                    print(f"schedule failed: {exc}", file=sys.stderr); return 1
                print(f"{job.name}: interval={job.interval_seconds:g}s")
        elif args.cmd == "rsi":
            rsi = core.rsi
            if args.action == "observe":
                o = rsi.observe()
                print(f"evals: {o.eval_scores}")
                print(f"weakest: {o.weakest_suite} failing: {list(o.failing_suites)} "
                      f"gate_denials: {o.gate_denials} failed_scheduler: {o.failed_scheduler_runs} "
                      f"episodic: {o.episodic_active} semantic: {o.semantic_active} backlog: {o.backlog}")
            elif args.action == "propose":
                made = rsi.propose()
                if not made: print("no proposal: no measurable weakness found")
                for p in made: print(f"proposal {p.id}: [{p.kind}] {p.target} — {p.rationale}")
            elif args.action == "pending":
                for p in rsi.pending():
                    print(f"proposal {p.id}: [{p.kind}] {p.target} patch={p.patch} score={p.score}")
            elif args.action in ("evaluate", "approve", "reject", "auto-adopt", "rollback"):
                if args.proposal_id is None:
                    print(f"rsi {args.action} requires a proposal id"); return 2
                try:
                    if args.action == "evaluate":
                        r = rsi.evaluate(args.proposal_id)
                        print(f"proposal {args.proposal_id}: {r.verdict} — {r.detail}")
                    elif args.action == "auto-adopt":
                        p = rsi.auto_adopt(args.proposal_id)
                        print(f"proposal {p.id}: {p.status} (config applied and persisted)")
                    elif args.action == "rollback":
                        p = rsi.rollback(args.proposal_id)
                        print(f"proposal {p.id}: {p.status} (previous value restored)")
                    else:
                        p = rsi.decide(args.proposal_id, "approved" if args.action == "approve" else "rejected")
                        print(f"proposal {p.id}: {p.status} (patch not applied; adoption is a separate owner step)")
                except (KeyError, ValueError) as exc:
                    print(f"rsi {args.action} failed: {exc}", file=sys.stderr); return 1
            elif args.action == "rollback-check":
                rolled = rsi.rollback_if_regressed()
                if not rolled:
                    print("rollback-check: no regression; no auto-adoption rolled back")
                for p in rolled:
                    print(f"proposal {p.id}: rolled_back (post-adoption regression)")
            elif args.action == "cycle":
                c = rsi.run_cycle()
                print(f"cycle: {c.status} proposals={list(c.proposals)} evaluations={c.evaluations} auto_adopted={list(c.auto_adopted)} — {c.detail}")
            elif args.action == "history":
                h = rsi.history()
                if not h["proposals"] and not h["cycles"]:
                    print("no RSI history yet: no proposals, no cycles")
                for p in h["proposals"]:
                    verdict = (p["score"] or {}).get("verdict")
                    print(f"proposal {p['proposal_id']}: [{p['kind']}] {p['target']} "
                          f"status={p['status']} verdict={verdict}")
                for c in h["cycles"]:
                    pl = c["payload"]
                    print(f"cycle: {pl.get('status')} proposals={pl.get('proposals')} "
                          f"evaluations={pl.get('evaluations')} auto_adopted={pl.get('auto_adopted')}")
            elif args.action == "submit":
                if not args.target or not args.rationale:
                    print("rsi submit requires --target and --rationale"); return 2
                try:
                    patch = json.loads(args.patch) if args.patch else None
                    p = rsi.submit(args.kind, args.target, args.rationale, patch)
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f"rsi submit failed: {exc}", file=sys.stderr); return 1
                print(f"proposal {p.id}: [{p.kind}] {p.target}")
        elif args.cmd == "cognition":
            from .gate import Risk
            cog = core.cognition
            try:
                risk = Risk[args.risk.upper()] if args.risk else None
            except KeyError:
                print(f"unknown risk class {args.risk!r}", file=sys.stderr); return 2
            if args.action == "query":
                if args.subject is None:
                    print("cognition query requires an action"); return 2
                q = cog.self_query(args.subject, goal=args.goal, where=args.where,
                                   risk=risk or Risk.READ, approved=args.approved,
                                   authority=args.authority, schedule=args.schedule)
                print(f"who: {q.who['name']} v{q.who['version']} "
                      f"({q.who['lineage']}, provider {q.who['provider']}, authority {q.who['authority']})")
                print(f"what: {q.what}")
                print(f"why: {q.why if q.why is not None else '(no answerable why — blocked)'}")
                print(f"where: {q.where}")
                print(f"how: risk={q.how['risk']} approved={int(q.how['approved'])}")
                print(f"when: schedule={q.when['schedule']}")
                print(f"blocked: {int(q.blocked)} (ledger SELF_QUERY entry {q.entry_id})")
            elif args.action == "deliberate":
                if args.subject is None or args.options is None or args.evidence is None \
                        or args.chosen is None or not args.reason:
                    print("cognition deliberate requires an action, --options, --evidence, --chosen, and --reason"); return 2
                try:
                    d = cog.deliberate(args.subject, json.loads(args.options),
                                       json.loads(args.evidence), args.chosen,
                                       args.reason, risk=risk or Risk.EXTERNAL)
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f"cognition deliberate failed: {exc}", file=sys.stderr); return 1
                print(f"deliberation {d.id}: {d.action} -> {d.chosen} ({d.reason})")
            elif args.action == "approve":
                if args.subject is None:
                    print("cognition approve requires an action"); return 2
                d = cog.request_approval(args.subject, risk or Risk.EXTERNAL,
                                         deliberation_id=args.deliberation_id,
                                         approved=args.approved)
                print(f"allowed: {int(d.allowed)} risk={d.risk.value} reason: {d.reason}")
            elif args.action == "unknown-record":
                if args.subject is None:
                    print("cognition unknown-record requires a question"); return 2
                try:
                    u = cog.record_unknown(args.subject, context=args.context)
                except ValueError as exc:
                    print(f"cognition unknown-record failed: {exc}", file=sys.stderr); return 1
                print(f"unknown {u.id}: {u.question}")
            elif args.action == "unknowns":
                rows = cog.unknowns()
                if not rows:
                    print("no known unknowns recorded")
                for u in rows:
                    print(f"unknown {u.id}: [{u.status}] {u.question}")
            elif args.action == "unknown-resolve":
                if args.subject is None or args.resolution_id is None:
                    print("cognition unknown-resolve requires an unknown id and --resolution-id"); return 2
                try:
                    u = cog.resolve_unknown(int(args.subject), args.resolution_id)
                except (KeyError, ValueError) as exc:
                    print(f"cognition unknown-resolve failed: {exc}", file=sys.stderr); return 1
                print(f"unknown {u.id}: resolved via memory item {u.resolution_id}")
            elif args.action == "metacognition":
                m = cog.metacognition()
                print(f"evals: {m.eval_scores}")
                print(f"failing: {list(m.failing_suites)} semantic: {m.semantic_count} "
                      f"mean_confidence={m.mean_semantic_confidence} "
                      f"corrected={m.corrected_semantic} "
                      f"overconfident={m.overconfident_corrections} "
                      f"calibration_gap={m.calibration_gap} "
                      f"gate_denials={m.gate_denials} open_unknowns={m.open_unknowns}")
                print(f"assessment: {m.assessment}")
            elif args.action == "check":
                r = cog.comprehension_check()
                print(f"comprehension: checked={r.checked} mismatches={len(r.mismatches)}")
                for mm in r.mismatches:
                    print(f"  episodic {mm.episodic_id} -> semantic {mm.semantic_id}: {mm.detail}")
            elif args.action == "scheduled-check":
                print(cog.scheduled_self_query())
        elif args.cmd == "episode":
            from .episode import _today
            if args.action == "list":
                records = orch.episode_list()
                if not records:
                    print("no episodes yet: compose one with 'episode compose'")
                for r in records:
                    print(f"{r.day}: {r.document_path} (memory item {r.memory_id})")
            elif args.action == "show":
                day = args.day or _today()
                try:
                    found = core.episode.get(day)
                except ValueError as exc:
                    print(f"episode show failed: {exc}", file=sys.stderr); return 1
                if found is None:
                    print(f"no episode for {day}: compose one with 'episode compose --day {day}'", file=sys.stderr); return 1
                print(found.text)
            else:  # compose
                try:
                    made = orch.episode_compose(args.day)
                except ValueError as exc:
                    print(f"episode compose failed: {exc}", file=sys.stderr); return 1
                if made.already_existed:
                    print(f"episode for {made.day} already exists: {made.document_path} (memory item {made.memory_id})")
                else:
                    print(f"episode for {made.day}: {made.document_path} "
                          f"(memory item {made.memory_id}, ledger entry {made.ledger_entry_id})")
        elif args.cmd == "predict":
            pred = core.prediction
            try:
                if args.action == "predict":
                    if args.subject is None:
                        print("predict predict requires a stream"); return 2
                    p = pred.predict(args.subject, context=args.context)
                    if p.predicted is None:
                        print(f"stream {p.stream}: no prediction yet (no history)")
                    else:
                        print(f"stream {p.stream}: predicted {p.predicted!r} "
                              f"(p={p.probability:.3f}, entropy={p.entropy_bits:.3f} bits, "
                              f"prediction {p.id})")
                elif args.action == "observe":
                    if args.subject is None or args.actual is None:
                        print("predict observe requires a stream and --actual"); return 2
                    o = pred.observe(args.subject, args.actual,
                                     prediction_id=args.prediction_id,
                                     context=args.context)
                    surprise = ("no history" if o.surprise_bits is None
                                else f"{o.surprise_bits:.3f} bits")
                    print(f"outcome {o.id}: {o.actual!r} (predicted {o.predicted!r}, "
                          f"surprise {surprise}, surprising={int(o.surprising)})")
                elif args.action == "summary":
                    s = pred.summary(args.subject)
                    mean = ("none" if s["mean_surprise_bits"] is None
                            else f"{s['mean_surprise_bits']:.3f} bits")
                    print(f"outcomes={s['outcomes']} mean_surprise={mean} "
                          f"surprising={s['surprising']} learned={s['learned']} "
                          f"unlearned={s['unlearned']} "
                          f"promoted_to_unknowns={s['promoted_to_unknowns']}")
                elif args.action == "queue":
                    rows = pred.learning_queue(limit=args.limit)
                    if not rows:
                        print("learning queue empty: nothing unlearned")
                    for o in rows:
                        surprise = ("no history" if o.surprise_bits is None
                                    else f"{o.surprise_bits:.3f} bits")
                        print(f"outcome {o.id}: [{o.stream}] {o.actual!r} "
                              f"surprise {surprise}")
                elif args.action == "learn":
                    if args.subject is None:
                        print("predict learn requires an outcome id"); return 2
                    o = pred.mark_learned(int(args.subject))
                    print(f"outcome {o.id}: learned")
                elif args.action == "promote":
                    made = pred.promote_to_unknowns(
                        limit=args.limit, threshold_bits=args.threshold)
                    if not made:
                        print("no surprising outcomes to promote")
                    for p in made:
                        print(f"outcome {p['outcome_id']}: unknown {p['unknown_id']}")
                elif args.action == "review":
                    r = pred.scheduled_surprise_review()
                    print(f"review: promoted unknowns {r['promoted_unknowns']} "
                          f"outcomes={r['summary']['outcomes']} "
                          f"surprising={r['summary']['surprising']}")
            except (KeyError, ValueError, RuntimeError) as exc:
                print(f"predict {args.action} failed: {exc}", file=sys.stderr); return 1
        elif args.cmd == "population":
            pop = core.population
            if args.action == "list":
                elites = pop.elites()
                if not elites:
                    print("population archive empty: run one with 'population run --variation ...'")
                for e in elites:
                    print(f"elite {e.id}: cell={e.cell} offspring={e.offspring_id} "
                          f"generation={e.generation} quality={e.quality:g} "
                          f"eval_passes={e.eval_passes} variation={e.variation}")
                print(f"diversity: {pop.diversity()} occupied cell(s)")
            elif args.action == "run":
                if not args.variation:
                    print("population run requires at least one --variation JSON"); return 2
                try:
                    variations = [json.loads(v) for v in args.variation]
                    heldout = None
                    if args.heldout:
                        with open(args.heldout, encoding="utf-8") as f:
                            heldout = json.load(f)
                    run = pop.run_population(variations, heldout_items=heldout)
                except (ValueError, json.JSONDecodeError, OSError) as exc:
                    print(f"population run failed: {exc}", file=sys.stderr); return 1
                for r in run.results:
                    print(f"variation {r.variation}: {r.verdict} "
                          f"quality={r.quality} archived={int(r.archived)} — {r.detail}")
                print(f"population: {run.archived_count()} archived, "
                      f"{run.cells_occupied} cell(s) occupied")
            else:  # adopt
                if args.elite_id is None:
                    print("population adopt requires an elite id"); return 2
                try:
                    e = pop.adopt_elite(args.elite_id, approved=args.approved)
                except (KeyError, ValueError) as exc:
                    print(f"population adopt failed: {exc}", file=sys.stderr); return 1
                print(f"elite {e.id} adopted: cell={e.cell} genome={e.genome}")
        elif args.cmd == "synthesize":
            syn = core.synthesis
            if args.action == "list":
                tools = syn.synthesized()
                if not tools:
                    print("no synthesized tools kept: propose one with 'synthesize propose ...'")
                for t in tools:
                    print(f"{t.name}: {t.description} ({len(t.cases)} case(s))")
            elif args.action == "remove":
                if args.name is None:
                    print("synthesize remove requires a tool name"); return 2
                try:
                    syn.remove(args.name)
                except KeyError as exc:
                    print(f"synthesize remove failed: {exc}", file=sys.stderr); return 1
                print(f"synthesized tool {args.name} removed")
            else:  # propose
                if args.name is None or not args.description:
                    print("synthesize propose requires a name and --description"); return 2
                try:
                    if args.code_file:
                        with open(args.code_file, encoding="utf-8") as f:
                            code = f.read()
                    elif args.code is not None:
                        code = args.code
                    else:
                        print("synthesize propose requires --code or --code-file"); return 2
                    if args.cases_file:
                        with open(args.cases_file, encoding="utf-8") as f:
                            cases = json.load(f)
                    elif args.cases is not None:
                        cases = json.loads(args.cases)
                    else:
                        print("synthesize propose requires --cases or --cases-file"); return 2
                    proposal = syn.propose(args.name, args.description, code,
                                           cases, timeout_seconds=args.timeout)
                except (ValueError, json.JSONDecodeError, OSError) as exc:
                    print(f"synthesize propose failed: {exc}", file=sys.stderr); return 1
                print(f"tool {proposal.name}: {proposal.verdict} — {proposal.detail}")
                return 0 if proposal.verdict == "accepted" else 1
        elif args.cmd == "ecosystem":
            eco = core.ecosystem
            if args.action == "list":
                try:
                    members = orch.ecosystem_list(status=args.status)
                except ValueError as exc:
                    print(f"ecosystem list failed: {exc}", file=sys.stderr); return 1
                if not members:
                    print("no ecosystem members yet: register one with 'ecosystem register <manifest.json> --approved'")
                for m in members:
                    print(f"{m.name}: kind={m.kind} status={m.status} "
                          f"capabilities={list(m.capabilities)} "
                          f"permissions={[r.value for r in m.permissions]}")
            elif args.action == "register":
                if args.subject is None:
                    print("ecosystem register requires a manifest JSON file"); return 2
                try:
                    with open(args.subject, encoding="utf-8") as f:
                        manifest = json.load(f)
                    m = orch.ecosystem_register(manifest, approved=args.approved)
                except (ValueError, PermissionError, json.JSONDecodeError, OSError) as exc:
                    print(f"ecosystem register failed: {exc}", file=sys.stderr); return 1
                print(f"member {m.name}: kind={m.kind} status={m.status} "
                      f"permissions={[r.value for r in m.permissions]}")
            elif args.action == "inspect":
                if args.subject is None:
                    print("ecosystem inspect requires a member name"); return 2
                try:
                    m = orch.ecosystem_inspect(args.subject)
                except KeyError as exc:
                    print(f"ecosystem inspect failed: {exc}", file=sys.stderr); return 1
                print(f"{m.name}: kind={m.kind} status={m.status}")
                print(f"capabilities: {list(m.capabilities)}")
                print(f"permissions: {[r.value for r in m.permissions]}")
                print(f"description: {m.description}")
                print(f"metadata: {json.dumps(m.metadata, sort_keys=True)}")
                if m.status == "retired":
                    print(f"retire_reason: {m.retire_reason}")
            else:  # retire
                if args.subject is None:
                    print("ecosystem retire requires a member name"); return 2
                try:
                    m = orch.ecosystem_retire(args.subject, reason=args.reason,
                                              approved=args.approved)
                except (KeyError, ValueError, PermissionError) as exc:
                    print(f"ecosystem retire failed: {exc}", file=sys.stderr); return 1
                print(f"member {m.name}: {m.status}")
        elif args.cmd == "toolpack":
            mgr = core.toolpacks
            if args.action == "list":
                packs = mgr.loaded_packs()
                if not packs:
                    print("no toolpacks loaded: load one with 'toolpack load <pack_dir> --approved'")
                for pk in packs:
                    print(f"{pk.name} v{pk.version}: tools={pk.tool_names}")
            elif args.action == "load":
                if args.path is None:
                    print("toolpack load requires a pack directory"); return 2
                try:
                    rec = mgr.load_toolpack(args.path, approved=args.approved)
                except (ValueError, PermissionError, FileNotFoundError) as exc:
                    print(f"toolpack load failed: {exc}", file=sys.stderr); return 1
                print(f"toolpack {rec.name} loaded: tools={rec.tools}")
            else:  # unload
                if args.path is None:
                    print("toolpack unload requires a pack name"); return 2
                try:
                    mgr.unload_toolpack(args.path, approved=args.approved)
                except (ValueError, PermissionError) as exc:
                    print(f"toolpack unload failed: {exc}", file=sys.stderr); return 1
                print(f"toolpack {args.path} unloaded")
        elif args.cmd == "serve":
            from .server import serve
            serve(core, host=args.host, port=args.port)
        elif args.cmd == "offspring":
            off = core.offspring
            if args.action == "list":
                records = off.offspring()
                if not records:
                    print("no offspring yet: spawn one with 'offspring spawn --variation ...'")
                for r in records:
                    verdict = (r.score or {}).get("verdict")
                    print(f"offspring {r.id}: generation {r.generation} "
                          f"status={r.status} variation={r.variation} "
                          f"verdict={verdict} sandbox={r.sandbox_dir}")
            elif args.action == "spawn":
                if args.variation is None:
                    print("offspring spawn requires --variation JSON"); return 2
                try:
                    variation = json.loads(args.variation)
                    r = off.spawn(variation, sandbox_dir=args.sandbox_dir,
                                  seed_limit=args.seed_limit,
                                  seed_min_confidence=args.seed_min_confidence)
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f"offspring spawn failed: {exc}", file=sys.stderr); return 1
                print(f"offspring {r.id}: generation {r.generation} "
                      f"seed_hash={r.seed_hash[:16]} sandbox={r.sandbox_dir}")
            elif args.action in ("evaluate", "promote", "retire"):
                if args.offspring_id is None:
                    print(f"offspring {args.action} requires an offspring id"); return 2
                try:
                    if args.action == "evaluate":
                        heldout = None
                        if args.heldout:
                            with open(args.heldout, encoding="utf-8") as f:
                                heldout = json.load(f)
                        ev = off.evaluate(args.offspring_id, heldout_items=heldout)
                        print(f"offspring {args.offspring_id}: {ev.verdict} — {ev.detail}")
                    elif args.action == "promote":
                        r = off.promote(args.offspring_id, approved=args.approved)
                        print(f"offspring {r.id}: {r.status} (genome adopted; parent ledger continues)")
                    else:
                        r = off.retire(args.offspring_id, reason=args.reason)
                        print(f"offspring {r.id}: {r.status}")
                except (KeyError, ValueError, OSError) as exc:
                    print(f"offspring {args.action} failed: {exc}", file=sys.stderr); return 1
        elif args.cmd == "backup":
            import time as _time
            from .backup import create_backup
            out = args.out or _time.strftime("steelbound-backup-%Y%m%d-%H%M%S.zip",
                                             _time.localtime())
            try:
                summary = create_backup(core.settings.data_dir, out, conn=core.conn)
            except (ValueError, OSError) as exc:
                print(f"backup failed: {exc}", file=sys.stderr); return 1
            core.ledger.append("cli", "BACKUP_CREATED", {
                "path": out, "db_sha256": summary["db_sha256"],
                "ledger_entries": summary["ledger_entries"],
                "files": summary["files"]})
            print(f"backed up to {summary['path']}: "
                  f"{summary['ledger_entries']} ledger entries, "
                  f"{summary['files']} workspace files")
        elif args.cmd == "export":
            from .exporter import export_bundle
            text = json.dumps(export_bundle(core), indent=2)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f: f.write(text + "\n")
                print(f"exported bundle to {args.out}")
            else:
                print(text)
        elif args.cmd == "import":
            from .exporter import import_bundle
            try:
                with open(args.path, encoding="utf-8") as f: bundle = json.load(f)
                summary = import_bundle(core, bundle)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                print(f"import rejected: {exc}", file=sys.stderr); return 1
            print(f"imported: {summary['ledger_imported']} ledger entries, "
                  f"{summary['memory_imported']} memory items, "
                  f"{summary['proposals_imported']} proposals; chain valid: True")
    finally: core.close()
    return 0

if __name__ == "__main__": sys.exit(main())

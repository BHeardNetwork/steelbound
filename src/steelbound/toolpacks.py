"""Toolpack SDK — file-based plugin packs for the tool registry.

A *toolpack* is a directory a third party (or the owner) can drop on
disk and load into a running core without touching Steelbound's own
source:

    mypack/
        pack.json       manifest: name, version, description, tools
        helpers.py      one or more Python modules
        ...

Each manifest tool declares a name, a description, a *declared* risk,
an entrypoint ``"module.py:function"``, and an argument schema. Loading
a pack is an all-or-nothing, fail-closed pipeline:

1. Gate: loading is a WRITE_LOCAL action (``toolpack_load``) and
   unloading is a WRITE_LOCAL action (``toolpack_unload``); both go
   through the core's gate and fail closed without approval.
2. Manifest: ``pack.json`` must parse and validate (names, risks,
   entrypoints, schemas). Anything malformed refuses the whole pack.
3. Static validation: *every* ``*.py`` file in the pack is checked
   with :func:`validate_pack_code` — synthesis-grade rules adapted
   from :mod:`steelbound.synthesis` for multi-function modules —
   *before* anything is imported. One bad file refuses the whole pack.
4. Risk resolution: a pack may never smuggle in more power than the
   loader allows (see "Risk semantics" below). A refused risk refuses
   the whole pack; risks are never silently downgraded.
5. Import + resolve + register: modules are imported from their file
   locations under unique module names, every entrypoint callable is
   resolved, and only then are the tools registered — so a pack is
   never partially registered.

Persistence: like synthesized tools, a loaded pack is recorded in the
``toolpacks`` sqlite table (name, version, manifest, pack directory,
tool names). :func:`load_toolpacks` re-loads every recorded pack at
Core startup from its directory — re-validating the sources and
refusing a pack whose manifest changed on disk since it was gated in.
Unloading deletes the row, unregisters the tools, and drops the
imported modules.

Risk semantics (the loader ceiling)
-----------------------------------
Risks are ordered READ < WRITE_LOCAL < EXTERNAL < DESTRUCTIVE <
FORBIDDEN. The manager has a *ceiling* (default WRITE_LOCAL):

* declared risk at or below the ceiling → the tool loads at its
  declared risk;
* declared FORBIDDEN → the tool, and therefore the whole pack, is
  refused unconditionally. No loader setting overrides this;
* declared risk above the ceiling (EXTERNAL / DESTRUCTIVE under the
  default ceiling) → the pack is refused *unless* the loader allows
  elevation — ``allow_elevated=True`` on the manager or on the load
  call — **and** that load call is owner-approved (``approved=True``).
  An elevated tool keeps its true declared risk, so every call to it
  still goes through the gate at that risk.

Execution model (v1 tradeoff)
-----------------------------
Pack callables run *in-process*, unlike synthesized tools, which run
in a subprocess sandbox on every call. The reasons: a pack is a
multi-function module with shared helpers and ordinary Python return
values (synthesis tools are a single pure function with
JSON-serializable returns), and re-serialising arbitrary pack state
through a subprocess per call is not faithful. The mitigations are
the static validation below (allowlisted imports only, no dunder or
private attribute access, no dangerous builtins, no import-time side
effects beyond definitions and constant assignments) plus the same
gate + ledger every tool call passes through.

Honest limits: static validation is *not* a sandbox. A validated pack
runs with the core process's privileges for anything the allowlist
permits (arithmetic, string and collection work, ``math``/``re``/
``json`` and friends) and can still burn CPU or memory. Only load
packs you trust, exactly as you would a Python dependency; subprocess
isolation for packs is future work — see docs/TOOLPACKS.md.

Ecosystem hook: when the core carries an ``ecosystem`` registry, a
loaded pack is registered as a member of kind ``"toolpack"`` and an
unloaded pack's member is retired. The hook is duck-typed and
best-effort (guarded with ``getattr``; this module never imports the
ecosystem module): a missing registry is skipped silently, and a
registry refusal — e.g. a retired member name that cannot be reused —
never breaks tool loading.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field

from .gate import Risk
from .tools import Tool

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# More restrictive == larger number. FORBIDDEN is not on the scale a
# loader may ever grant; it exists so comparisons fail closed.
_RISK_ORDER = {
    Risk.READ: 0,
    Risk.WRITE_LOCAL: 1,
    Risk.EXTERNAL: 2,
    Risk.DESTRUCTIVE: 3,
    Risk.FORBIDDEN: 4,
}

#: Default loader ceiling: packs load READ and WRITE_LOCAL tools
#: freely (once the load itself is approved); anything riskier needs
#: explicit elevation — see the module docstring.
DEFAULT_CEILING = Risk.WRITE_LOCAL

# Stdlib modules a pack module may import: pure computation only.
# Deliberately absent: os, sys, subprocess, io, pathlib, socket,
# urllib, importlib, ctypes, and anything else that can reach the
# filesystem, the network, the interpreter, or another process.
ALLOWED_IMPORTS = frozenset({
    "bisect", "collections", "datetime", "decimal", "fractions",
    "functools", "heapq", "itertools", "json", "math", "operator",
    "random", "re", "statistics", "string", "textwrap", "typing",
})

# Builtins a pack module may never name — the same set synthesis
# bans, for the same reason: each one reaches outside pure
# computation (I/O, evaluation, or an introspection escape).
_BANNED_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "help",
    "breakpoint", "exit", "quit", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "memoryview", "object", "type",
})

_MODULE_COUNTER = itertools.count()


class ToolpackError(ValueError):
    """A pack is malformed, unsafe, or otherwise refused. Subclasses
    ValueError so callers that already treat bad tool definitions as
    ValueError (synthesis' convention) handle packs the same way."""


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

def validate_pack_code(code: str, source: str = "<pack module>") -> None:
    """Static gate for one pack module's source. Raises ToolpackError
    on any shape a pack module must not have; passing here does not
    run the module — it only means the module is *importable* under
    the pack rules.

    Mirrors ``synthesis.validate_code`` (same banned names, same
    dunder rule) adapted for real modules: several functions and
    classes are allowed, and imports are allowed from an allowlist
    instead of being banned outright. Module level may hold only a
    docstring, allowlisted imports, function/class definitions, and
    assignments — no bare calls or control flow — so importing a pack
    cannot execute arbitrary statements.
    """
    if not isinstance(code, str) or not code.strip():
        raise ToolpackError(f"{source}: module source must be a non-empty string")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolpackError(f"{source}: module does not parse: {exc}")
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Assign, ast.AnnAssign)):
            continue
        raise ToolpackError(
            f"{source}: module level may hold only imports, definitions"
            " and assignments (no import-time statements)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ToolpackError(
                        f"{source}: import of {alias.name!r} is not allowed"
                        " (allowlist only)")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ToolpackError(
                    f"{source}: relative imports are not allowed")
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise ToolpackError(
                    f"{source}: import from {node.module!r} is not allowed"
                    " (allowlist only)")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ToolpackError(
                    f"{source}: private/dunder attribute access"
                    f" ({node.attr!r}) is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise ToolpackError(
                    f"{source}: {node.id!r} is not allowed in pack code")
            if node.id.startswith("__"):
                raise ToolpackError(
                    f"{source}: dunder name {node.id!r} is not allowed")
        elif isinstance(node, ast.Global):
            raise ToolpackError(
                f"{source}: 'global' statements are not allowed in pack code")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risk: Risk  # the *declared* risk, before loader risk resolution
    entrypoint_file: str
    entrypoint_func: str
    schema: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PackManifest:
    name: str
    version: str
    description: str
    tools: tuple[ToolSpec, ...]
    raw: dict


def _parse_risk(value, where: str) -> Risk:
    if isinstance(value, Risk):
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        for risk in Risk:
            if wanted in (risk.value, risk.name.lower()):
                return risk
    raise ToolpackError(f"{where}: unknown risk {value!r}")


def _parse_entrypoint(value, where: str) -> tuple[str, str]:
    if not isinstance(value, str) or ":" not in value:
        raise ToolpackError(
            f"{where}: entrypoint must look like 'module.py:function'")
    file_part, func_part = (part.strip() for part in value.rsplit(":", 1))
    if not file_part.endswith(".py") or not func_part:
        raise ToolpackError(
            f"{where}: entrypoint must look like 'module.py:function'")
    if os.path.isabs(file_part) or ".." in file_part.split("/"):
        raise ToolpackError(
            f"{where}: entrypoint file must stay inside the pack directory")
    if not all(part.isidentifier() for part in func_part.split(".")):
        raise ToolpackError(
            f"{where}: entrypoint function {func_part!r} is not a valid name")
    return file_part, func_part


def parse_manifest(data) -> PackManifest:
    """Validate decoded pack.json content. Raises ToolpackError."""
    if not isinstance(data, dict):
        raise ToolpackError("pack.json must be a JSON object")
    name = data.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ToolpackError(
            f"pack name {name!r} must match ^[a-z][a-z0-9_]*$")
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ToolpackError(f"pack {name!r} needs a non-empty 'version'")
    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ToolpackError(f"pack {name!r} needs a non-empty 'description'")
    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ToolpackError(f"pack {name!r} needs a non-empty 'tools' list")
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_tools):
        where = f"pack {name!r} tool {i}"
        if not isinstance(entry, dict):
            raise ToolpackError(f"{where}: tool entry must be an object")
        tname = entry.get("name")
        if not isinstance(tname, str) or not _NAME_RE.match(tname):
            raise ToolpackError(
                f"{where}: tool name {tname!r} must match ^[a-z][a-z0-9_]*$")
        if tname in seen:
            raise ToolpackError(f"pack {name!r}: duplicate tool name {tname!r}")
        seen.add(tname)
        tdesc = entry.get("description")
        if not isinstance(tdesc, str) or not tdesc.strip():
            raise ToolpackError(f"tool {tname!r} needs a non-empty 'description'")
        risk = _parse_risk(entry.get("risk", "read"), f"tool {tname!r}")
        efile, efunc = _parse_entrypoint(entry.get("entrypoint"),
                                         f"tool {tname!r}")
        schema = {}
        for key in ("schema", "args_schema", "parameters", "args"):
            if key in entry:
                schema = entry[key]
                break
        if not isinstance(schema, dict):
            raise ToolpackError(f"tool {tname!r}: 'schema' must be an object")
        specs.append(ToolSpec(tname, tdesc.strip(), risk, efile, efunc,
                              dict(schema)))
    return PackManifest(name, version.strip(), description.strip(),
                        tuple(specs), data)


def read_manifest(pack_dir: str) -> PackManifest:
    """Read and validate ``<pack_dir>/pack.json``. Raises ToolpackError."""
    path = os.path.join(pack_dir, "pack.json")
    if not os.path.isfile(path):
        raise ToolpackError(f"no pack.json in {pack_dir!r}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolpackError(f"pack.json in {pack_dir!r} is unreadable: {exc}")
    return parse_manifest(data)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class LoadedPack:
    name: str
    version: str
    description: str
    pack_dir: str
    tool_names: list[str]
    risks: dict  # tool name -> effective Risk
    module_names: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolpackLoadResult:
    name: str
    version: str
    tools: list[str]
    pack_dir: str
    risks: dict  # tool name -> effective risk value string

    @property
    def pack_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Ecosystem hook (duck-typed, best-effort; never imports ecosystem code)
# ---------------------------------------------------------------------------

def _ecosystem_manifest(record: LoadedPack) -> dict:
    permissions = sorted({r.value for r in record.risks.values()},
                         key=lambda v: _RISK_ORDER[Risk(v)])
    return {
        "name": record.name,
        "kind": "toolpack",
        "capabilities": list(record.tool_names),
        "permissions": permissions,
        "description": record.description,
        "metadata": {
            "version": record.version,
            "pack_dir": record.pack_dir,
            "tools": list(record.tool_names),
        },
    }


def _ecosystem_register(core, record: LoadedPack, approved: bool) -> None:
    """Register a loaded pack as an ecosystem member of kind
    'toolpack'. Best-effort by design: no core attribute, no registry
    method, an already-present member, and a registry refusal are all
    non-fatal — the tools are the deliverable, membership is a view."""
    if core is None:
        return
    eco = getattr(core, "ecosystem", None)
    if eco is None:
        return
    try:
        inspect = getattr(eco, "inspect", None)
        if callable(inspect):
            try:
                member = inspect(record.name)
            except Exception:
                member = None
            if member is not None:
                # Already a member (e.g. persisted across a restart).
                # A retired member's name is not reusable in the
                # registry, so there is nothing to do either way.
                return
        manifest = _ecosystem_manifest(record)
        register = getattr(eco, "register", None)
        if callable(register):
            try:
                register(manifest, approved=approved)
            except TypeError:
                try:
                    register(manifest)
                except Exception:
                    pass
            except Exception:
                pass
            return
        for attr in ("register_member", "add_member", "enroll"):
            fn = getattr(eco, attr, None)
            if callable(fn):
                try:
                    fn(manifest)
                except TypeError:
                    try:
                        fn(**manifest)
                    except Exception:
                        pass
                except Exception:
                    pass
                return
    except Exception:
        return


def _ecosystem_retire(core, record: LoadedPack, approved: bool) -> None:
    """Retire an unloaded pack's ecosystem member. Best-effort, like
    registration: a missing or refusing registry never blocks unload."""
    if core is None:
        return
    eco = getattr(core, "ecosystem", None)
    if eco is None:
        return
    try:
        retire = getattr(eco, "retire", None)
        if callable(retire):
            try:
                retire(record.name, reason="toolpack unloaded",
                       approved=approved)
            except TypeError:
                try:
                    retire(record.name)
                except Exception:
                    pass
            except Exception:
                pass
            return
        for attr in ("retire_member", "remove_member", "unregister"):
            fn = getattr(eco, attr, None)
            if callable(fn):
                try:
                    fn(record.name)
                except Exception:
                    pass
                return
    except Exception:
        return


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@dataclass
class _Prepared:
    manifest: PackManifest
    pack_dir: str  # absolute
    files: dict  # relative posix path -> absolute path


class ToolpackManager:
    """Loads, tracks, persists, and unloads toolpacks for one core.

    ``allow_elevated`` is the loader posture for above-ceiling risks;
    it can also be granted per load call. Either way, elevation only
    ever takes effect on an owner-approved load — see the module
    docstring for the full risk semantics.
    """

    def __init__(self, ledger, conn: sqlite3.Connection, tools=None,
                 gate=None, allow_elevated: bool = False,
                 ceiling: Risk = DEFAULT_CEILING, core=None):
        self.ledger = ledger
        self.conn = conn
        # The wired construction in config.py passes only ledger, conn
        # and core=; fall back to the core's registry and gate so a
        # manager built that way is fully functional, not inert.
        if tools is None and core is not None:
            tools = getattr(core, "tools", None)
        if gate is None and core is not None:
            gate = getattr(core, "gate", None)
        self.tools = tools
        self.gate = gate
        self.allow_elevated = bool(allow_elevated)
        # FORBIDDEN can never be a grantable ceiling; clamp it to the
        # highest grantable risk rather than honouring it.
        self.ceiling = Risk.DESTRUCTIVE if ceiling is Risk.FORBIDDEN else ceiling
        self.core = core
        self._loaded: dict[str, LoadedPack] = {}
        conn.execute("""CREATE TABLE IF NOT EXISTS toolpacks(
            name TEXT PRIMARY KEY, version TEXT NOT NULL,
            description TEXT NOT NULL, pack_dir TEXT NOT NULL,
            manifest TEXT NOT NULL, tool_names TEXT NOT NULL,
            ts REAL NOT NULL)""")
        conn.commit()

    # -- introspection ----------------------------------------------------

    def loaded_packs(self) -> list[LoadedPack]:
        return [self._loaded[name] for name in sorted(self._loaded)]

    def get_pack(self, name: str) -> LoadedPack | None:
        if name in self._loaded:
            return self._loaded[name]
        row = self.conn.execute(
            "SELECT name,version,description,pack_dir,manifest,tool_names"
            " FROM toolpacks WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        return LoadedPack(row[0], row[1], row[2], row[3],
                          json.loads(row[5]), {}, [],
                          json.loads(row[4]))

    def persisted_packs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name,version,description,pack_dir,manifest,tool_names,ts"
            " FROM toolpacks ORDER BY name ASC").fetchall()
        return [{"name": r[0], "version": r[1], "description": r[2],
                 "pack_dir": r[3], "manifest": json.loads(r[4]),
                 "tool_names": json.loads(r[5]), "ts": r[6]} for r in rows]

    # -- gate / ledger helpers ----------------------------------------------

    def _require_gate(self, action: str, approved: bool) -> None:
        if self.gate is not None:
            decision = self.gate.evaluate(action, Risk.WRITE_LOCAL,
                                          approved=approved)
            if not decision.allowed:
                raise PermissionError(decision.reason)
        elif not approved:
            raise PermissionError(f"{action} requires approval.")

    def _ledger_refused(self, pack_name, pack_dir: str, stage: str,
                        reason: str) -> None:
        if self.ledger:
            self.ledger.append("toolpacks", "TOOLPACK_REFUSED", {
                "pack": pack_name, "pack_dir": pack_dir,
                "stage": stage, "reason": reason})

    # -- preparation: manifest, validation, collisions, risks ---------------

    def _prepare(self, pack_dir: str, manifest: PackManifest) -> _Prepared:
        abs_dir = os.path.abspath(pack_dir)
        files: dict[str, str] = {}
        for root, dirnames, filenames in os.walk(abs_dir):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for filename in filenames:
                if filename.endswith(".py"):
                    full = os.path.join(root, filename)
                    rel = os.path.relpath(full, abs_dir).replace(os.sep, "/")
                    files[rel] = full
        # Validate EVERY module before anything is imported.
        for rel in sorted(files):
            try:
                with open(files[rel], encoding="utf-8") as f:
                    source = f.read()
            except OSError as exc:
                raise ToolpackError(f"module {rel!r} is unreadable: {exc}")
            validate_pack_code(source, source=rel)
        for spec in manifest.tools:
            if spec.entrypoint_file not in files:
                raise ToolpackError(
                    f"tool {spec.name!r}: entrypoint module"
                    f" {spec.entrypoint_file!r} is not in the pack")
        return _Prepared(manifest, abs_dir, files)

    def _check_collisions(self, manifest: PackManifest,
                          reloading: str | None = None) -> None:
        if manifest.name in self._loaded and reloading != manifest.name:
            raise ToolpackError(
                f"toolpack {manifest.name!r} is already loaded")
        if reloading != manifest.name and self.conn.execute(
                "SELECT 1 FROM toolpacks WHERE name=?",
                (manifest.name,)).fetchone() is not None:
            raise ToolpackError(
                f"toolpack {manifest.name!r} is already loaded")
        if self.tools is not None:
            for spec in manifest.tools:
                if self.tools.get(spec.name) is not None:
                    raise ToolpackError(
                        f"tool name {spec.name!r} collides with an"
                        " existing tool")

    def _resolve_risks(self, manifest: PackManifest, ceiling: Risk,
                       allow_elevated: bool, approved: bool) -> dict:
        """Declared risk -> effective risk, or ToolpackError. Risks
        are refused, never downgraded: an above-ceiling tool loads
        only under elevation + approval, at its true declared risk."""
        effective: dict[str, Risk] = {}
        refused: list[str] = []
        for spec in manifest.tools:
            declared = spec.risk
            if declared is Risk.FORBIDDEN:
                refused.append(
                    f"{spec.name} declares FORBIDDEN risk (never loadable)")
            elif _RISK_ORDER[declared] > _RISK_ORDER[ceiling]:
                if allow_elevated and approved:
                    effective[spec.name] = declared
                else:
                    refused.append(
                        f"{spec.name} declares {declared.value} risk, above"
                        f" the loader ceiling {ceiling.value} (needs an"
                        " elevated loader and an approved load)")
            else:
                effective[spec.name] = declared
        if refused:
            raise ToolpackError(
                f"pack {manifest.name!r} refused on risk: "
                + "; ".join(refused))
        return effective

    # -- activation: import, resolve, register --------------------------------

    def _activate(self, prepared: _Prepared, risks: dict) -> LoadedPack:
        manifest = prepared.manifest
        modules: dict[str, object] = {}
        sys_names: list[str] = []
        try:
            needed = sorted({s.entrypoint_file for s in manifest.tools})
            for rel in needed:
                unique = "steelbound_toolpack_{}_{}_{}".format(
                    re.sub(r"[^a-z0-9_]", "_", manifest.name),
                    re.sub(r"[^a-z0-9_]", "_",
                           os.path.splitext(os.path.basename(rel))[0]),
                    next(_MODULE_COUNTER))
                spec = importlib.util.spec_from_file_location(
                    unique, prepared.files[rel])
                if spec is None or spec.loader is None:
                    raise ToolpackError(f"cannot import module {rel!r}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[unique] = module
                sys_names.append(unique)
                spec.loader.exec_module(module)
                modules[rel] = module
            resolved = []
            for tspec in manifest.tools:
                target = modules[tspec.entrypoint_file]
                for part in tspec.entrypoint_func.split("."):
                    try:
                        target = getattr(target, part)
                    except AttributeError:
                        raise ToolpackError(
                            f"tool {tspec.name!r}: entrypoint"
                            f" {tspec.entrypoint_func!r} not found in"
                            f" {tspec.entrypoint_file!r}")
                if not callable(target):
                    raise ToolpackError(
                        f"tool {tspec.name!r}: entrypoint"
                        f" {tspec.entrypoint_func!r} is not callable")
                resolved.append((tspec, target))
        except ToolpackError:
            for name in sys_names:
                sys.modules.pop(name, None)
            raise
        except Exception as exc:
            for name in sys_names:
                sys.modules.pop(name, None)
            raise ToolpackError(
                f"pack {manifest.name!r} failed to import:"
                f" {type(exc).__name__}: {exc}")
        # Every entrypoint resolved: registration is now all-or-nothing
        # in practice — plain dict inserts that cannot fail part-way
        # after the collision checks in _check_collisions.
        if self.tools is not None:
            for tspec, fn in resolved:
                self.tools.register(Tool(
                    tspec.name, tspec.description, risks[tspec.name],
                    _make_pack_fn(fn), dict(tspec.schema)))
        record = LoadedPack(
            manifest.name, manifest.version, manifest.description,
            prepared.pack_dir, [s.name for s in manifest.tools],
            dict(risks), list(sys_names), manifest.raw)
        self._loaded[record.name] = record
        return record

    def _persist(self, record: LoadedPack) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO toolpacks"
            "(name,version,description,pack_dir,manifest,tool_names,ts)"
            " VALUES(?,?,?,?,?,?,?)",
            (record.name, record.version, record.description,
             record.pack_dir,
             json.dumps(record.manifest, sort_keys=True),
             json.dumps(record.tool_names), time.time()))
        self.conn.commit()

    # -- load / unload ---------------------------------------------------------

    def load_toolpack(self, pack_dir, approved: bool = False, *,
                      allow_elevated: bool = False,
                      ceiling: Risk | None = None) -> ToolpackLoadResult:
        """Gate, validate, and register one pack directory. Atomic:
        any failure leaves the registry, the table, and sys.modules
        exactly as they were, and is ledgered TOOLPACK_REFUSED."""
        pack_dir_str = str(pack_dir)
        ceiling = self.ceiling if ceiling is None else ceiling
        if ceiling is Risk.FORBIDDEN:
            ceiling = Risk.DESTRUCTIVE
        allow_elevated = self.allow_elevated or allow_elevated
        fallback_name = os.path.basename(os.path.abspath(pack_dir_str))
        try:
            self._require_gate("toolpack_load", approved)
        except PermissionError as exc:
            self._ledger_refused(fallback_name, pack_dir_str, "gate",
                                 str(exc))
            raise
        manifest: PackManifest | None = None
        stage = "manifest"
        try:
            if not os.path.isdir(pack_dir_str):
                raise ToolpackError(
                    f"pack directory {pack_dir_str!r} does not exist")
            manifest = read_manifest(pack_dir_str)
            stage = "validation"
            prepared = self._prepare(pack_dir_str, manifest)
            self._check_collisions(manifest)
            stage = "risk"
            risks = self._resolve_risks(manifest, ceiling,
                                         allow_elevated, approved)
            stage = "import"
            record = self._activate(prepared, risks)
        except ToolpackError as exc:
            self._ledger_refused(
                manifest.name if manifest else fallback_name,
                pack_dir_str, stage, str(exc))
            raise
        self._persist(record)
        if self.ledger:
            self.ledger.append("toolpacks", "TOOLPACK_LOADED", {
                "pack": record.name, "version": record.version,
                "pack_dir": record.pack_dir,
                "tools": list(record.tool_names),
                "risks": {k: v.value for k, v in record.risks.items()}})
        _ecosystem_register(self.core, record, approved=approved)
        return ToolpackLoadResult(
            record.name, record.version, list(record.tool_names),
            record.pack_dir,
            {k: v.value for k, v in record.risks.items()})

    # Short alias, convenient for CLI wiring.
    load = load_toolpack

    def unload_toolpack(self, pack_name: str,
                        approved: bool = False) -> LoadedPack:
        """Unregister a pack's tools, drop its modules, delete its
        persistence row, and retire its ecosystem member."""
        try:
            self._require_gate("toolpack_unload", approved)
        except PermissionError as exc:
            self._ledger_refused(pack_name, "", "gate", str(exc))
            raise
        record = self.get_pack(pack_name)
        if record is None:
            raise KeyError(f"no toolpack {pack_name!r} is loaded")
        if self.tools is not None:
            for tool_name in record.tool_names:
                self.tools.unregister(tool_name)
        for module_name in record.module_names:
            sys.modules.pop(module_name, None)
        self._loaded.pop(pack_name, None)
        self.conn.execute("DELETE FROM toolpacks WHERE name=?",
                          (pack_name,))
        self.conn.commit()
        if self.ledger:
            self.ledger.append("toolpacks", "TOOLPACK_UNLOADED", {
                "pack": pack_name, "tools": list(record.tool_names)})
        _ecosystem_retire(self.core, record, approved=approved)
        return record

    unload = unload_toolpack

    # -- startup reload ---------------------------------------------------------

    def reload_all(self) -> list[str]:
        """Re-load every persisted pack from its directory, as at
        Core startup. No gate: the original load was gated, and the
        row in the table is its record. A pack whose manifest changed
        on disk, whose sources no longer validate, or whose tools now
        collide is skipped and ledgered TOOLPACK_REFUSED — a reload
        failure never aborts startup."""
        try:
            rows = self.conn.execute(
                "SELECT name,pack_dir,manifest FROM toolpacks"
                " ORDER BY name ASC").fetchall()
        except sqlite3.Error:
            return []
        loaded: list[str] = []
        for name, pack_dir, manifest_json in rows:
            if name in self._loaded:
                loaded.append(name)
                continue
            try:
                stored = json.loads(manifest_json)
                manifest = read_manifest(pack_dir)
                if manifest.raw != stored:
                    raise ToolpackError(
                        "pack manifest changed on disk since it was"
                        " loaded; refusing to reload")
                prepared = self._prepare(pack_dir, manifest)
                self._check_collisions(manifest, reloading=name)
                risks = {s.name: s.risk for s in manifest.tools}
                if any(r is Risk.FORBIDDEN for r in risks.values()):
                    raise ToolpackError(
                        "pack declares a FORBIDDEN tool; refusing"
                        " to reload")
                record = self._activate(prepared, risks)
            except Exception as exc:
                self._ledger_refused(name, str(pack_dir), "reload",
                                     str(exc))
                continue
            _ecosystem_register(self.core, record, approved=True)
            loaded.append(name)
        return loaded


def _make_pack_fn(fn):
    """The live callable for a pack tool: a thin in-process wrapper
    (see the module docstring for why v1 runs in-process). Calls
    still pass through ToolRegistry.run, hence through the gate and
    into the ledger, exactly like built-in tools."""
    def _call(**kwargs):
        return fn(**kwargs)
    _call.__name__ = getattr(fn, "__name__", "toolpack_tool")
    return _call


# ---------------------------------------------------------------------------
# Core-level convenience API (the wiring surface for config.py / cli.py)
# ---------------------------------------------------------------------------

def _manager_for(core) -> ToolpackManager:
    """Find the core's manager, or build and attach one. A manager
    wired in by Core construction (``core.toolpacks`` or
    ``core.toolpack_manager``) always wins."""
    for attr in ("toolpacks", "toolpack_manager"):
        existing = getattr(core, attr, None)
        if isinstance(existing, ToolpackManager):
            existing.core = core
            return existing
    manager = ToolpackManager(
        getattr(core, "ledger", None), core.conn,
        getattr(core, "tools", None), getattr(core, "gate", None),
        core=core)
    try:
        core.toolpack_manager = manager
    except Exception:
        pass
    return manager


def load_toolpack(core, pack_dir, approved: bool = False, *,
                  allow_elevated: bool = False,
                  ceiling: Risk | None = None) -> ToolpackLoadResult:
    """Load one pack directory into a core. See ToolpackManager."""
    return _manager_for(core).load_toolpack(
        pack_dir, approved=approved, allow_elevated=allow_elevated,
        ceiling=ceiling)


def unload_toolpack(core, pack_name: str,
                    approved: bool = False) -> LoadedPack:
    """Unload a previously loaded pack from a core."""
    return _manager_for(core).unload_toolpack(pack_name,
                                              approved=approved)


def load_toolpacks(core) -> list[str]:
    """Startup reload: re-register every persisted pack on a core.
    Returns the names of the packs now loaded."""
    return _manager_for(core).reload_all()


def list_toolpacks(core) -> list[LoadedPack]:
    """The packs currently loaded on a core."""
    return _manager_for(core).loaded_packs()

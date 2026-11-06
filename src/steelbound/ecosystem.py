"""Ecosystem registry: the members that make up the wider system
built around a Steelbound core.

The core does not stand alone. Around it sit applications, offspring
cores, toolpacks, and services — the *members* of its ecosystem. This
module is the registry of those members: who they are, what they can
do, and what they are allowed to do. It generalises the rule that has
bound offspring cores from the start (docs/OFFSPRING.md): a member's
permissions are always a subset of its host's, never broader, and
everything significant is ledgered.

Permission ceiling — the hard invariant:

* The default ceiling is every :class:`~steelbound.gate.Risk` class
  except ``FORBIDDEN``. ``FORBIDDEN`` is never grantable: a manifest
  that requests it is refused outright, and it is stripped from any
  configured ceiling, so no configuration can make it grantable.
* An engine may be constructed with a narrower ``ceiling`` (a member-
  hosting core may itself be bounded — an offspring core, for
  example, should not register members more powerful than itself).
  A member's permissions must be a subset of the effective ceiling.
* Subset enforcement is by **rejection, never silent clamping**: a
  manifest asking for a permission outside the ceiling raises
  ``ValueError`` and registers nothing. The manifest that is stored
  and ledgered is always exactly the manifest that was asked for and
  approved — the registry never quietly rewrites a member's powers
  to something narrower than it declared, because a member running
  under different permissions than it believes it has is worse than
  no member at all.

Gating and ledgering:

* Registration is a ``WRITE_LOCAL`` gate action
  (``ecosystem_register``); retirement is ``DESTRUCTIVE``
  (``ecosystem_retire``). Both go through ``Gate.evaluate`` with the
  caller's ``approved`` flag passed through, and a denied operation
  raises ``PermissionError`` and mutates nothing. (The gate itself
  ledgering its ``GATE_DECISION`` is the gate's own record, as with
  every other gated action in this codebase.)
* Every successful mutation is ledgered: ``ECOSYSTEM_REGISTER`` with
  the full manifest in the payload, ``ECOSYSTEM_RETIRE`` with the
  member name and reason. Reads (list / inspect / act checks) are
  not ledgered, matching the rest of the codebase.

Act guard: a member may act only while it is registered and active.
``can_act`` answers False and ``require_active`` raises for a retired
or unknown member alike — the check fails closed, so a member that
was retired (or never existed) cannot act under any circumstances.

Members persist in the engine's own ``ecosystem_members`` table, so
the registry survives a Core reopen exactly as it was left. No
network is used anywhere in this module.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from .gate import Gate, Risk

MEMBER_KINDS = ("app", "offspring", "toolpack", "service")
MEMBER_STATUSES = ("active", "retired")

# The core's permission ceiling: every risk class except FORBIDDEN.
# FORBIDDEN is not a power a member can hold; it is the class of
# actions that are never allowed at all.
DEFAULT_CEILING = frozenset(r for r in Risk if r is not Risk.FORBIDDEN)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _parse_risk(value) -> Risk:
    """One permission entry: a Risk, or its name/value as a string
    (``"write_local"`` / ``"WRITE_LOCAL"``), as JSON manifests carry."""
    if isinstance(value, Risk):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        for risk in Risk:
            if key in (risk.name.lower(), risk.value):
                return risk
    raise ValueError(f"unknown permission {value!r}")


def normalize_permissions(permissions, ceiling=None) -> tuple[Risk, ...]:
    """Validate a permission list against a ceiling. Returns the
    permissions as a de-duplicated tuple of Risk, in declaration
    order. Raises ValueError on an unknown permission, on FORBIDDEN
    (refused outright, whatever the ceiling), and on any permission
    outside the ceiling (rejected, never clamped)."""
    if permissions is None:
        permissions = []
    if isinstance(permissions, (str, Risk)):
        permissions = [permissions]
    if not isinstance(permissions, (list, tuple, set, frozenset)):
        raise ValueError("permissions must be a list of risk classes")
    effective = DEFAULT_CEILING if ceiling is None else _normalize_ceiling(ceiling)
    parsed: list[Risk] = []
    for value in permissions:
        risk = _parse_risk(value)
        if risk is Risk.FORBIDDEN:
            raise ValueError(
                "permission 'forbidden' is refused outright: FORBIDDEN "
                "is never a grantable permission")
        if risk not in effective:
            raise ValueError(
                f"permission {risk.value!r} exceeds the ecosystem "
                f"permission ceiling {sorted(r.value for r in effective)}: "
                "member permissions must be a subset of the host's")
        if risk not in parsed:
            parsed.append(risk)
    return tuple(parsed)


def _normalize_ceiling(ceiling) -> frozenset:
    """A configured ceiling as a frozenset of Risk. FORBIDDEN is
    always removed: no configuration can make it grantable."""
    if ceiling is None:
        return DEFAULT_CEILING
    if isinstance(ceiling, (str, Risk)):
        ceiling = [ceiling]
    parsed = {_parse_risk(v) for v in ceiling}
    parsed.discard(Risk.FORBIDDEN)
    return frozenset(parsed)


@dataclass(frozen=True)
class MemberManifest:
    """A member declaration, as data. ``register`` also accepts a
    plain dict with these fields (what a JSON manifest file holds)."""
    name: str
    kind: str
    capabilities: tuple = ()
    permissions: tuple = ()
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "permissions": [p.value if isinstance(p, Risk) else p
                            for p in self.permissions],
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EcosystemMember:
    """One registered member, as stored in the registry."""
    name: str
    kind: str
    capabilities: tuple
    permissions: tuple  # tuple[Risk, ...]
    description: str
    metadata: dict
    status: str  # active | retired
    registered_ts: float
    updated_ts: float
    retired_ts: float | None = None
    retire_reason: str = ""

    def manifest(self) -> dict:
        """The manifest this member was registered with."""
        return {
            "name": self.name,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict:
        out = self.manifest()
        out.update({
            "status": self.status,
            "registered_ts": self.registered_ts,
            "updated_ts": self.updated_ts,
            "retired_ts": self.retired_ts,
            "retire_reason": self.retire_reason,
        })
        return out


def validate_manifest(manifest, ceiling=None) -> dict:
    """Normalise a manifest (dict or MemberManifest) to a plain dict
    with permissions as a tuple of Risk. Raises ValueError on any
    field a member declaration must not have — including any
    permission outside the ceiling. Nothing is registered here; this
    is pure validation, shared by ``register`` and direct callers."""
    if isinstance(manifest, MemberManifest):
        data = manifest.to_dict()
    elif isinstance(manifest, EcosystemMember):
        data = manifest.manifest()
    elif isinstance(manifest, dict):
        data = dict(manifest)
    else:
        raise ValueError("manifest must be a JSON object")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("manifest needs a non-empty 'name'")
    name = name.strip()
    kind = data.get("kind")
    if kind not in MEMBER_KINDS:
        raise ValueError(
            f"manifest 'kind' must be one of {list(MEMBER_KINDS)}, "
            f"got {kind!r}")
    capabilities = data.get("capabilities") or []
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    if not isinstance(capabilities, (list, tuple)) or not all(
            isinstance(c, str) and c.strip() for c in capabilities):
        raise ValueError("'capabilities' must be a list of non-empty strings")
    capabilities = tuple(c.strip() for c in capabilities)
    permissions = normalize_permissions(data.get("permissions") or [], ceiling)
    description = data.get("description") or ""
    if not isinstance(description, str):
        raise ValueError("'description' must be a string")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("'metadata' must be a JSON object")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError):
        raise ValueError("'metadata' must be JSON-serializable")
    return {
        "name": name,
        "kind": kind,
        "capabilities": capabilities,
        "permissions": permissions,
        "description": description.strip(),
        "metadata": dict(metadata),
    }


class EcosystemEngine:
    def __init__(self, ledger, conn: sqlite3.Connection, core=None,
                 ceiling=None):
        self.ledger = ledger
        self.conn = conn
        self.core = core
        # The effective ceiling this engine registers under. Narrower
        # than DEFAULT_CEILING when the hosting core is itself bounded.
        self.ceiling = _normalize_ceiling(ceiling)
        self._fallback_gate = Gate(ledger)
        conn.execute("""CREATE TABLE IF NOT EXISTS ecosystem_members(
            name TEXT PRIMARY KEY, kind TEXT NOT NULL,
            capabilities TEXT NOT NULL, permissions TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active',
            registered_ts REAL NOT NULL, updated_ts REAL NOT NULL,
            retired_ts REAL, retire_reason TEXT NOT NULL DEFAULT '')""")
        conn.commit()

    @property
    def gate(self) -> Gate:
        """The host core's gate when wired into a Core (so its
        auto-approve flag and ledger are the live ones); otherwise a
        gate over this engine's own ledger. Resolved per call, so a
        gate swapped or re-flagged on the core is honoured."""
        core_gate = getattr(self.core, "gate", None)
        return core_gate if core_gate is not None else self._fallback_gate

    # -- records -----------------------------------------------------------
    def _row_to_member(self, row) -> EcosystemMember:
        return EcosystemMember(
            name=row[0], kind=row[1],
            capabilities=tuple(json.loads(row[2])),
            permissions=tuple(Risk(p) for p in json.loads(row[3])),
            description=row[4], metadata=json.loads(row[5]),
            status=row[6], registered_ts=row[7], updated_ts=row[8],
            retired_ts=row[9], retire_reason=row[10])

    def _get_row(self, name: str):
        return self.conn.execute(
            "SELECT name,kind,capabilities,permissions,description,metadata,"
            " status,registered_ts,updated_ts,retired_ts,retire_reason"
            " FROM ecosystem_members WHERE name=?", (name,)).fetchone()

    def list_members(self, status: str | None = None) -> list[EcosystemMember]:
        """Every registered member, by name; optionally only those
        with the given status (``active`` or ``retired``)."""
        if status is not None and status not in MEMBER_STATUSES:
            raise ValueError(
                f"status must be one of {list(MEMBER_STATUSES)}, got {status!r}")
        if status is None:
            rows = self.conn.execute(
                "SELECT name,kind,capabilities,permissions,description,"
                " metadata,status,registered_ts,updated_ts,retired_ts,"
                " retire_reason FROM ecosystem_members ORDER BY name ASC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT name,kind,capabilities,permissions,description,"
                " metadata,status,registered_ts,updated_ts,retired_ts,"
                " retire_reason FROM ecosystem_members WHERE status=?"
                " ORDER BY name ASC", (status,)).fetchall()
        return [self._row_to_member(r) for r in rows]

    def inspect(self, name: str) -> EcosystemMember:
        """One member by name, active or retired. Raises KeyError for
        an unknown name — inspection of what does not exist is an
        error, never an empty record."""
        row = self._get_row(name)
        if row is None:
            raise KeyError(f"no ecosystem member {name!r}")
        return self._row_to_member(row)

    # -- act guard -----------------------------------------------------------
    def can_act(self, name: str) -> bool:
        """True only for a registered, active member. A retired or
        unknown member cannot act: the check fails closed."""
        row = self._get_row(name)
        return row is not None and row[6] == "active"

    def require_active(self, name: str) -> EcosystemMember:
        """The member, if it may act. Raises KeyError for an unknown
        member and PermissionError for a retired one."""
        member = self.inspect(name)
        if member.status != "active":
            raise PermissionError(
                f"ecosystem member {name!r} is {member.status} and cannot act")
        return member

    # -- registration ---------------------------------------------------------
    def register(self, manifest, approved: bool = False) -> EcosystemMember:
        """Register a member from its manifest. The manifest is fully
        validated first — kind, capabilities, and above all the
        permission-subset rule — so an invalid or over-powered
        manifest is refused before the gate is even consulted and
        registers nothing. Registration itself is a WRITE_LOCAL gate
        action: without approval (and without the gate's local-write
        auto-approval) it raises PermissionError and mutates nothing.
        A name that is already registered — active or retired — is
        refused: names are never reused, so a retired member's record
        can never be silently replaced by a new member."""
        data = validate_manifest(manifest, self.ceiling)
        if self._get_row(data["name"]) is not None:
            raise ValueError(
                f"an ecosystem member named {data['name']!r} already exists")
        decision = self.gate.evaluate(
            "ecosystem_register", Risk.WRITE_LOCAL, approved=approved)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        now = time.time()
        self.conn.execute(
            "INSERT INTO ecosystem_members(name,kind,capabilities,permissions,"
            " description,metadata,status,registered_ts,updated_ts)"
            " VALUES(?,?,?,?,?,?, 'active', ?, ?)",
            (data["name"], data["kind"], _canonical(list(data["capabilities"])),
             _canonical([p.value for p in data["permissions"]]),
             data["description"], _canonical(data["metadata"]), now, now))
        self.conn.commit()
        member = self.inspect(data["name"])
        if self.ledger:
            self.ledger.append("ecosystem", "ECOSYSTEM_REGISTER", {
                "name": member.name,
                "manifest": member.manifest(),
            })
        return member

    # -- retirement -------------------------------------------------------------
    def retire(self, name: str, reason: str = "",
               approved: bool = False) -> EcosystemMember:
        """Retire a member: it stays in the registry as a record but
        can never act again. Retirement is DESTRUCTIVE at the gate —
        without explicit owner approval it raises PermissionError and
        the member stays active. Retiring an already-retired member
        raises ValueError; an unknown name raises KeyError."""
        member = self.inspect(name)
        if member.status != "active":
            raise ValueError(
                f"ecosystem member {name!r} is already {member.status}")
        if not isinstance(reason, str):
            raise ValueError("retire reason must be a string")
        decision = self.gate.evaluate(
            "ecosystem_retire", Risk.DESTRUCTIVE, approved=approved)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        now = time.time()
        self.conn.execute(
            "UPDATE ecosystem_members SET status='retired', updated_ts=?,"
            " retired_ts=?, retire_reason=? WHERE name=?",
            (now, now, reason, name))
        self.conn.commit()
        retired = self.inspect(name)
        if self.ledger:
            self.ledger.append("ecosystem", "ECOSYSTEM_RETIRE", {
                "name": retired.name, "kind": retired.kind,
                "reason": reason,
            })
        return retired

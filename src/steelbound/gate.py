"""Deliberation gate: consequential actions are classified, checked, and
approved before execution. Fails closed: unknown actions are denied."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Risk(Enum):
    READ = "read"            # no side effects
    WRITE_LOCAL = "write_local"   # reversible, local
    EXTERNAL = "external"    # leaves the machine / contacts others
    DESTRUCTIVE = "destructive"   # hard or impossible to undo
    FORBIDDEN = "forbidden"  # never allowed (credentials, payments without explicit flow)

@dataclass(frozen=True)
class GateDecision:
    allowed: bool; risk: Risk; reason: str; needs_approval: bool

class Gate:
    def __init__(self, ledger=None, auto_approve_local_writes: bool = False):
        self.ledger = ledger; self.auto_approve_local_writes = auto_approve_local_writes

    def evaluate(self, action: str, risk: Risk, approved: bool = False) -> GateDecision:
        if risk is Risk.FORBIDDEN:
            d = GateDecision(False, risk, "Forbidden action class.", True)
        elif risk is Risk.READ:
            d = GateDecision(True, risk, "Read-only action.", False)
        elif risk is Risk.WRITE_LOCAL:
            ok = approved or self.auto_approve_local_writes
            d = GateDecision(ok, risk, "Local write approved." if ok else "Local write requires approval.", not ok)
        elif risk in (Risk.EXTERNAL, Risk.DESTRUCTIVE):
            d = GateDecision(bool(approved), risk,
                             f"{risk.value} action approved by owner." if approved else f"{risk.value} action requires explicit owner approval.", not approved)
        else:
            d = GateDecision(False, risk, "Unknown risk class: denied (fail closed).", True)
        if self.ledger:
            self.ledger.append("gate", "GATE_DECISION", {"action": action, "risk": risk.value, "allowed": d.allowed})
        return d

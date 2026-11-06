"""Tool registry. A tool exists only if it declares its schema, risk, and
side effects. Execution always passes through the gate and into the ledger."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from .gate import Gate, Risk

@dataclass
class Tool:
    name: str; description: str; risk: Risk; fn: object; schema: dict = field(default_factory=dict)

class ToolRegistry:
    def __init__(self, gate: Gate, ledger=None):
        self.gate = gate; self.ledger = ledger; self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None: self._tools[tool.name] = tool
    def unregister(self, name: str) -> None: self._tools.pop(name, None)
    def list(self) -> list[Tool]: return list(self._tools.values())
    def get(self, name: str) -> Tool | None: return self._tools.get(name)

    def run(self, name: str, args: dict, approved: bool = False):
        tool = self._tools.get(name)
        if tool is None: raise KeyError(f"unknown tool: {name}")
        decision = self.gate.evaluate(f"tool:{name}", tool.risk, approved=approved)
        if not decision.allowed:
            if self.ledger: self.ledger.append("tools", "TOOL_DENIED", {"tool": name, "reason": decision.reason})
            raise PermissionError(decision.reason)
        result = tool.fn(**(args or {}))
        if self.ledger: self.ledger.append("tools", "TOOL_RUN", {"tool": name})
        return result

def make_file_tools(workspace: str) -> list[Tool]:
    root = os.path.abspath(workspace); os.makedirs(root, exist_ok=True)
    def _safe(path: str) -> str:
        p = os.path.abspath(os.path.join(root, path))
        if os.path.commonpath([p, root]) != root: raise PermissionError("path escapes the sandbox workspace")
        return p
    def read_file(path: str) -> str:
        with open(_safe(path), encoding="utf-8") as f: return f.read()
    def write_file(path: str, content: str) -> str:
        p = _safe(path); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f: f.write(content)
        return f"wrote {len(content)} chars to {path}"
    def list_files() -> list[str]:
        out = []
        for dp, _, fns in os.walk(root):
            for fn in fns: out.append(os.path.relpath(os.path.join(dp, fn), root))
        return sorted(out)
    return [
        Tool("read_file", "Read a file inside the sandbox workspace.", Risk.READ, read_file, {"path": "str"}),
        Tool("write_file", "Write a file inside the sandbox workspace.", Risk.WRITE_LOCAL, write_file, {"path": "str", "content": "str"}),
        Tool("list_files", "List files in the sandbox workspace.", Risk.READ, list_files, {}),
    ]

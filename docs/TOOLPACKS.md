# Toolpacks — the Steelbound plugin SDK

A **toolpack** is a directory of Python code that adds tools to a
Steelbound core without modifying Steelbound itself. Packs are
loaded at runtime, gated like every other consequential action,
statically validated before anything is imported, risk-clamped by
the loader, recorded in the append-only ledger, persisted so they
survive restarts, and registered as members of the ecosystem
registry (kind `toolpack`) while loaded.

Implementation: `src/steelbound/toolpacks.py` (`ToolpackManager`,
plus the module-level `load_toolpack` / `unload_toolpack` /
`load_toolpacks` helpers). Tests: `tests/test_toolpacks.py`.
Working example: `examples/toolpacks/text_stats/`.

---

## 1. Pack layout

```
my_pack/
    pack.json        required manifest
    tools_a.py       one or more Python modules
    helpers.py       every *.py file in the tree is validated
    sub/             subdirectories are scanned too
        more.py
```

- Modules may not import each other (imports are allowlist-only —
  see §4). Share code by putting related entrypoints in one module,
  or duplicate the small helper.
- `__pycache__` directories are ignored.

## 2. Manifest schema (`pack.json`)

```json
{
  "name": "text_stats",
  "version": "1.0.0",
  "description": "Pure text statistics.",
  "tools": [
    {
      "name": "text_stats",
      "description": "Count words, characters, sentences and estimate reading time.",
      "risk": "read",
      "entrypoint": "textstats.py:text_stats",
      "schema": { "text": "str" }
    }
  ]
}
```

| Field | Required | Rules |
|---|---|---|
| `name` | yes | `^[a-z][a-z0-9_]*$` |
| `version` | yes | non-empty string (semver recommended, not enforced) |
| `description` | yes | non-empty string |
| `tools` | yes | non-empty list |
| `tools[].name` | yes | same name rule; unique within the pack; must not collide with an existing tool |
| `tools[].description` | yes | non-empty string |
| `tools[].risk` | no (default `read`) | one of `read`, `write_local`, `external`, `destructive`, `forbidden` (enum names like `READ` also accepted) — the *declared* risk, subject to §5 |
| `tools[].entrypoint` | yes | `"module.py:function"` — file relative to the pack root (may include subdirectories, never `..` or absolute paths); function may be dotted (`module.py:Class.method`) and must resolve to a callable |
| `tools[].schema` | no (default `{}`) | object describing arguments, e.g. `{"text": "str", "top_n": "int"}`. Informational: it is passed through to the registered `Tool`. (`args_schema`, `parameters`, `args` are accepted as aliases.) |

Anything malformed — missing file, invalid JSON, bad names, unknown
risk, duplicate tool names, a missing entrypoint module — refuses
the **whole pack** (see §6).

## 3. Lifecycle

**Loading is a gated, atomic pipeline.** `toolpack_load` is a
`WRITE_LOCAL` gate action: without owner approval (and without the
core's `auto_approve_local_writes` setting) the load is denied
before a single file is validated or imported.

1. **Gate** — `toolpack_load` evaluated at `WRITE_LOCAL`.
2. **Manifest** — `pack.json` parsed and validated (§2).
3. **Static validation** — *every* `*.py` file in the pack passes
   `validate_pack_code` (§4). Nothing has been imported yet.
4. **Collision checks** — pack not already loaded; no tool name
   collides with a registered tool.
5. **Risk resolution** — declared risks resolved against the loader
   ceiling (§5). A refused risk refuses the whole pack.
6. **Import + resolve** — each needed module is imported from its
   file location under a unique module name
   (`steelbound_toolpack_<pack>_<module>_<n>`); every entrypoint
   callable is resolved. Any failure cleans up `sys.modules` and
   registers nothing.
7. **Register + persist + ledger** — tools are registered in the
   core's `ToolRegistry` (so every call runs through the gate and
   ledger like a built-in tool), a row is written to the
   `toolpacks` sqlite table, `TOOLPACK_LOADED` is appended to the
   ledger, and the pack is registered as an ecosystem member (§7).

**Unloading** (`toolpack_unload`, also `WRITE_LOCAL`-gated)
unregisters the pack's tools, drops its imported modules from
`sys.modules`, deletes its persistence row, appends
`TOOLPACK_UNLOADED`, and retires its ecosystem member. Unloading an
unknown pack raises `KeyError`.

**Refusals** at any stage append `TOOLPACK_REFUSED` to the ledger
with the pack name, directory, stage (`gate` / `manifest` /
`validation` / `risk` / `import` / `reload`) and reason — and leave
the registry, the table, and `sys.modules` exactly as they were.
Loading is all-or-nothing: a pack is never partially registered.

**Persistence & startup reload.** The `toolpacks` table records
name, version, description, pack directory, the full manifest, and
tool names. `Core` construction wires `core.toolpacks` (a
`ToolpackManager`) and calls `load_toolpacks(core)`, which re-loads
every recorded pack from its directory: sources are re-validated,
and a pack whose on-disk manifest **differs from the manifest that
was gated in** is refused (ledgered, stage `reload`) rather than
silently reloaded — edits to a pack require an explicit unload +
load so they pass the gate again. A reload failure never aborts
Core startup.

## 4. Static validation rules

`validate_pack_code(code, source)` mirrors the philosophy of
`synthesis.validate_code` — same banned names, same dunder rule —
adapted for real multi-function modules. A module must:

- parse as Python;
- hold only a docstring, imports, function/class definitions, and
  assignments at module level — **no import-time statements** (no
  bare calls, loops, or conditionals), so importing a pack cannot
  execute arbitrary code;
- import only from the allowlist (checked everywhere, including
  imports inside functions; relative imports are banned):

  > `bisect`, `collections`, `datetime`, `decimal`, `fractions`,
  > `functools`, `heapq`, `itertools`, `json`, `math`, `operator`,
  > `random`, `re`, `statistics`, `string`, `textwrap`, `typing`

  Notably absent: `os`, `sys`, `subprocess`, `io`, `pathlib`,
  `socket`, `urllib`, `importlib`, `ctypes` — anything that can
  reach the filesystem, the network, the interpreter, or another
  process;
- never name a banned builtin: `open`, `eval`, `exec`, `compile`,
  `__import__`, `input`, `help`, `breakpoint`, `exit`, `quit`,
  `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`,
  `memoryview`, `object`, `type`;
- never access a private or dunder attribute (any attribute
  starting with `_`, e.g. `x.__class__`) or a dunder name;
- contain no `global` statements.

## 5. Risk semantics (the loader ceiling)

Risks are ordered `READ < WRITE_LOCAL < EXTERNAL < DESTRUCTIVE <
FORBIDDEN`. The manager has a **ceiling**, default `WRITE_LOCAL`,
and declared risks are **refused, never silently downgraded** —
clamping a tool's risk down would lie to the gate about what the
tool can do.

| Declared risk | Default loader | Elevated loader |
|---|---|---|
| `read`, `write_local` (≤ ceiling) | loads at declared risk | loads |
| `external`, `destructive` (> ceiling) | **pack refused** | loads at its true declared risk — only if the load call itself is owner-approved (`approved=True`) |
| `forbidden` | **pack refused** | **pack refused** — no setting overrides this |

- Elevation is granted by `allow_elevated=True` on the manager or
  on the individual load call; either way it takes effect only on
  an approved load.
- A custom ceiling narrows or widens the free band the same way:
  with `ceiling=Risk.READ`, even a `write_local` tool needs
  elevation. A `FORBIDDEN` ceiling is clamped to `DESTRUCTIVE` —
  forbidden can never be granted.
- An elevated tool keeps its declared risk in the registry, so
  **every call** to it still passes the gate at that risk (an
  `external` pack tool needs per-call approval to run).

## 6. API

```python
from steelbound.toolpacks import (
    ToolpackManager, ToolpackError, validate_pack_code,
    load_toolpack, unload_toolpack, load_toolpacks, list_toolpacks)
```

- `ToolpackManager(ledger, conn, tools=None, gate=None, *,
  allow_elevated=False, ceiling=Risk.WRITE_LOCAL, core=None)` —
  when `tools`/`gate` are omitted they are taken from `core`.
  `core.toolpacks` is the wired instance.
  - `.load_toolpack(pack_dir, approved=False, *,
    allow_elevated=False, ceiling=None) -> ToolpackLoadResult`
    (alias `.load`). The result carries `name` (alias
    `pack_name`), `version`, `tools` (list of tool names),
    `pack_dir`, and `risks` (tool name → effective risk value).
  - `.unload_toolpack(pack_name, approved=False) -> LoadedPack`
    (alias `.unload`)
  - `.loaded_packs()`, `.get_pack(name)`, `.persisted_packs()`,
    `.reload_all() -> list[str]`
- Module-level helpers take a core:
  `load_toolpack(core, pack_dir, approved=False, ...)`,
  `unload_toolpack(core, pack_name, approved=False)`,
  `load_toolpacks(core) -> list[str]` (startup reload),
  `list_toolpacks(core)`.
- Failures raise `ToolpackError` (a `ValueError`) for malformed,
  unsafe, or risk-refused packs; `PermissionError` for gate
  denials; `KeyError` for unloading an unknown pack.

### CLI

```
steelbound toolpack list
steelbound toolpack load <pack_dir> --approved
steelbound toolpack unload <pack_name> --approved
steelbound tool <tool_name> --args '{"text": "..."}'
```

## 7. Ecosystem hook

When the core carries an ecosystem registry (`core.ecosystem`, an
`EcosystemEngine`), a successful load registers the pack as a
member:

```json
{ "name": "<pack>", "kind": "toolpack",
  "capabilities": ["<tool>", "..."],
  "permissions": ["<effective risk values>"],
  "description": "<pack description>",
  "metadata": {"version": "...", "pack_dir": "...", "tools": [...]} }
```

and unload retires that member (`reason="toolpack unloaded"`).
The hook is duck-typed and best-effort — `toolpacks.py` never
imports the ecosystem module: a missing registry is skipped
silently, an already-present member (e.g. persisted across a
restart) is left as-is, and a registry refusal (a retired member's
name cannot be reused) never breaks tool loading or unloading.

## 8. Security model — and honest limits

**What protects the core**

- The load itself is owner-gated before anything is read beyond
  the manifest location.
- All-or-nothing validation of *every* module, before import:
  allowlisted imports only, no dangerous builtins, no dunder or
  private attribute access, no import-time side effects.
- Declared risks are resolved against a loader ceiling and never
  downgraded; `FORBIDDEN` is unconditionally un-loadable.
- Calls to pack tools go through the same gate + ledger as
  built-in tools, at the tool's effective risk.
- Startup reload re-validates and refuses packs whose manifest
  changed since the gated load.
- Every load, unload, and refusal is in the append-only ledger.

**Honest limits (v1)**

- Pack callables run **in-process**. Unlike synthesized tools —
  a single pure function with JSON-serializable returns, executed
  in a subprocess sandbox per call — a pack is a multi-function
  module with shared helpers and ordinary Python return values,
  and re-serialising arbitrary pack state through a subprocess per
  call would not be faithful. The mitigations are the static
  validation above plus the gate/ledger on every call.
- Static validation is **not a sandbox**. A validated pack runs
  with the core process's privileges for anything the allowlist
  permits (arithmetic, string/collection work, `math` / `re` /
  `json` and friends) and can still burn CPU or memory — there is
  no timeout or quota on pack calls in v1. Validation shrinks the
  surface; it does not make hostile code safe.
- **Only load packs you trust**, exactly as you would a Python
  dependency. Subprocess isolation for packs, call timeouts, and
  resource quotas are future work.

## 9. Worked example

`examples/toolpacks/text_stats/` ships two `READ` tools from one
module (`textstats.py`): `text_stats` (word/character/sentence
counts, unique words, reading-time estimate) and `word_frequency`.

```python
from steelbound.config import Settings, Core
from steelbound.toolpacks import load_toolpack, unload_toolpack

core = Core(Settings(data_dir=".steelbound"))
result = load_toolpack(core, "examples/toolpacks/text_stats",
                       approved=True)
print(result.tools)            # ['text_stats', 'word_frequency']

stats = core.tools.run(
    "text_stats",
    {"text": "The quick brown fox jumps. The lazy dog sleeps!"})
print(stats["words"], stats["sentences"])        # 9 2
print(stats["reading_time_minutes"])             # 0.045

core.tools.run("word_frequency",
               {"text": "a b a c a b", "top_n": 2})   # {'a': 3, 'b': 2}

unload_toolpack(core, "text_stats", approved=True)
```

Without `approved=True` the load raises `PermissionError`; the
tools above are `READ` risk, so once loaded they run without
per-call approval.

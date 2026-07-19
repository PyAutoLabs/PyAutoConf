# PyAutoConf — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

**This repo is the Nerves organ of the PyAuto organism** — the base
configuration/serialization layer (`autonerves`) that every scientific library
imports. It is a full organ (promoted to the seventh organ 2026-07); the
organism map below is generated from `PyAutoMind/repos.yaml`.

## The organism map

<!-- repos_sync:map:begin -->
**You are one organ of the PyAuto organism** — an agentic ecosystem for
human-led, natural-language software development. The organs below are
peer repositories; this repo is one of them, not a part of another.
Canonical boundaries live in `PyAutoBrain/ORGANISM.md`; the full body map
(every repo, not just organs) is `PyAutoMind/repos.yaml`.

| Organ | Repo | Role |
|-------|------|------|
| **Mind** | PyAutoMind | Intent, goals, priorities, workflow state; every task starts as a markdown prompt here. |
| **Brain** | PyAutoBrain | Reasoning/orchestration layer; how work is decomposed and routed; the specialist agents. |
| **Hands** | PyAutoHands | Packaging, tagging, notebook generation, PyPI release execution. |
| **Heart** | PyAutoHeart | Health/readiness — the authoritative "is it safe to release?" verdict. |
| **Memory** | PyAutoMemory | Long-term scientific/software/project knowledge (see science pointer below). |
| **Gut** | PyAutoGut | Owns the lifecycle of condemned self-material (stale branches, stashes, dead code/tests): holds it as durable, recoverable git refs through a transit window and voids it on a sweep. The storage mirror of Memory (retention vs release). |
| **Nerves** | PyAutoConf | The Nerves — the configuration/serialization layer connecting workspace conventions to libraries (layered config, version handshake, test_mode), delivered as the `autonerves` package. |

Call chain (always this order): **Brain → Heart (gate) → Build (execute)**. Brain agents are **conductors** (front-door; a human drives them; they decide *and* act) or **faculties** (read-only opinions the conductors consult; they judge and stop). New capability grows as a faculty, not a new organ, unless it owns state or effects no existing organ can.

Generated from `PyAutoMind/repos.yaml` + `PyAutoBrain/ORGANISM.md`; edit there, then run `python3 PyAutoMind/scripts/repos_sync.py --write`.
<!-- repos_sync:map:end -->

## What this repo is

**PyAutoConf** (package `autonerves`) is the configuration, serialization, and
I/O foundation of the PyAuto ecosystem: layered config with overrides,
dict/JSON/CSV serialization, FITS I/O, JSON-based priors, and the shared
`jax_wrapper` / `test_mode` utilities.

Dependency direction: autonerves is the **base layer**. It does **not** import
`autofit`, `autoarray`, `autogalaxy`, or `autolens`. All four of those depend
on autonerves, so any public-API change here ripples downstream.

## Related repos

- **Source siblings (downstream):** PyAutoFit, PyAutoArray, PyAutoGalaxy,
  PyAutoLens — all depend on autonerves.
- No `_workspace`, `_workspace_test`, or HowTo companion repo.
- No `docs/` / RTD site — the package source and `test_autonerves/` are the
  authoritative reference.

## Architecture

- `autonerves/conf.py` — layered config system (`Config` / `conf.instance`).
- `autonerves/dictable.py` — dict / JSON serialization (`output_to_json` / `from_json`).
- `autonerves/fitsable.py` — FITS I/O (`output_to_fits` / `ndarray_via_fits_from`).
- `autonerves/json_prior/` — JSON-based priors.
- `autonerves/tools/` — shared decorators and helpers.
- `test_autonerves/` — test suite.

## Quick commands

```bash
pip install -e ".[dev]"              # install with dev/test extras
python -m pytest test_autonerves/      # full test suite
python -m pytest test_autonerves/tools/test_decorators.py   # one focused test
black autonerves/                      # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autonerves/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Public API

The public surface is defined authoritatively in `autonerves/__init__.py` — read
it rather than trusting a hand-maintained list. Canonical import:

```python
from autonerves import conf
```

Key surfaces: `Config` / `conf.instance`, `output_to_json` / `from_json`
(`dictable.py`), `output_to_fits` / `ndarray_via_fits_from` (`fitsable.py`).

## Key rules / footguns

- All files use Unix line endings (LF, `\n`) — never `\r\n`.
- This is the base config/IO layer: a public-API change here can break
  PyAutoFit, PyAutoArray, PyAutoGalaxy, and PyAutoLens. Flag it loudly.
- YAML dict keys are lowercased on load (`muJy` → `mujy`); keep config keys and
  any matching Python registries snake_case-lowercase.

## Working on issues

1. Read the issue description and any linked plan.
2. Identify affected files and make the change.
3. Run the full suite: `python -m pytest test_autonerves/`.
4. If you changed public API, say so explicitly — PyAutoFit, PyAutoArray,
   PyAutoGalaxy, and PyAutoLens all depend on this package and may need updates.
5. Ensure all tests pass before opening a PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

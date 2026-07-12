# PyAutoConf — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

## What this repo is

**PyAutoConf** (package `autoconf`) is the configuration, serialization, and
I/O foundation of the PyAuto ecosystem: layered config with overrides,
dict/JSON/CSV serialization, FITS I/O, JSON-based priors, and the shared
`jax_wrapper` / `test_mode` utilities.

Dependency direction: autoconf is the **base layer**. It does **not** import
`autofit`, `autoarray`, `autogalaxy`, or `autolens`. All four of those depend
on autoconf, so any public-API change here ripples downstream.

## Related repos

- **Source siblings (downstream):** PyAutoFit, PyAutoArray, PyAutoGalaxy,
  PyAutoLens — all depend on autoconf.
- No `_workspace`, `_workspace_test`, or HowTo companion repo.
- No `docs/` / RTD site — the package source and `test_autoconf/` are the
  authoritative reference.

## Architecture

- `autoconf/conf.py` — layered config system (`Config` / `conf.instance`).
- `autoconf/dictable.py` — dict / JSON serialization (`output_to_json` / `from_json`).
- `autoconf/fitsable.py` — FITS I/O (`output_to_fits` / `ndarray_via_fits_from`).
- `autoconf/json_prior/` — JSON-based priors.
- `autoconf/tools/` — shared decorators and helpers.
- `test_autoconf/` — test suite.

## Quick commands

```bash
pip install -e ".[dev]"              # install with dev/test extras
python -m pytest test_autoconf/      # full test suite
python -m pytest test_autoconf/tools/test_decorators.py   # one focused test
black autoconf/                      # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autoconf/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Public API

The public surface is defined authoritatively in `autoconf/__init__.py` — read
it rather than trusting a hand-maintained list. Canonical import:

```python
from autoconf import conf
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
3. Run the full suite: `python -m pytest test_autoconf/`.
4. If you changed public API, say so explicitly — PyAutoFit, PyAutoArray,
   PyAutoGalaxy, and PyAutoLens all depend on this package and may need updates.
5. Ensure all tests pass before opening a PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
<!-- repos_sync:history:end -->

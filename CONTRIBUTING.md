# Contributing to Mathmethods MCP

Thanks for considering a contribution! This project is intentionally small and
simple — that is a feature, not a limitation. Keep the spirit:

- **Small beats clever.** Prefer a few readable lines over a clever one-liner.
- **Simple beats complex.** If a feature needs a dozen flags to explain, it is
  probably too much.
- **Correct first.** The math must be verifiable; style comes after.
- **Tested.** Every new tool or math change needs a test that proves the result.

## Setup

```bash
git clone https://github.com/agmonetti/mathmethods.git
cd mathmethods
uv sync --extra dev
uv run pytest
```

## Adding a tool

1. The math lives in `mathmethods/core/` (vendored from the upstream
   `modeladoYsimulacion-web` backend). If you add a method there, also fix the
   upstream copy and keep both in sync.
2. Tools are plain functions registered with `@mcp.tool()` in
   `mathmethods/server.py`. Write a clear docstring — the LLM picks the tool by
   that description.
3. Validate every numeric parameter with the existing `_clean_float` /
   `_clean_int` helpers and every expression with `compiler.validate`. Never
   bypass the lexical gate.
4. Add a test in `tests/` and run `uv run pytest` and `uv run ruff check .`.

## Rules

- No `eval`/`exec` on user input, ever. Route through `compiler.validate` /
  `safe_sympify`.
- Keep responses bounded: cap `N`, `n`, `steps`, grid sizes and downsample
  long arrays before returning.
- Commit with a conventional message (`feat:`, `fix:`, `docs:`, `chore:`).

## Keeping the vendored core in sync

The math core mirrors `modeladoYsimulacion-web/backend/app/methods/`. After
upstream changes, re-copy the files and rewrite `from app.core.utils import
...` to `from .utils import ...`. See the README "Keeping the vendored core in
sync" section.

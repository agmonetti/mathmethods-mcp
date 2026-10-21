# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-10-21

### Fixed

- **`root_bisection` — root at interval boundary**: when $f(a) = 0$ or $f(b) = 0$
  (within tolerance), bisection previously failed with `"f(a) y f(b) deben tener
  signos opuestos"`. Exact roots at the endpoints are now returned immediately with
  `num_iter: 0`.
- **ODE solvers (`ode_euler`, `ode_heun`, `ode_rk4`) — step desynchronization**:
  when $(x_f - x_0) / h$ was non-integer, the output grid $(x_n)$ and the integration
  step differed, causing desynchronization between table values and analytical errors.
  Solvers now use the effective step $h_{\text{eff}} = (x_f - x_0) / n_{\text{steps}}$
  (where $n_{\text{steps}} = \lceil (x_f - x_0) / h \rceil$) and expose `h_efectivo`
  in the output payload.
- **`dynamic_1d` & `core/utils.normalize_expr` — token-safe `sen` normalization**:
  bare-string `.replace("sen", "sin")` corrupted parameter or variable names
  containing `"sen"` (such as `"sense"` becoming `"sinse"`). Both locations now
  use word boundaries (`\bsen\b`).
- **`mc_hit_or_miss_1d` — confidence interval variance for signed functions**:
  for signed integrals, the estimator is the signed indicator ($p_+ - p_-$). The
  variance calculation previously used $p(1 - p)$ from unsigned hit-or-miss,
  producing ~12% narrower intervals on zero-crossing functions. The variance now
  correctly accounts for the signed indicator: $\mathrm{Var} = (p_+ + p_-) - (p_+ - p_-)^2$.
- **`integral_comparar` — validation requiring $n$ multiple of 6**: running with $n$
  incompatible with Simpson 1/3 (requires even $n$) or Simpson 3/8 (requires $n$
  multiple of 3) resulted in partial execution with per-method error keys. It now
  requires $n$ to be a multiple of 6 with an actionable hint (e.g. `Try n=6.`).

## [0.2.1] - 2026-10-20

### Fixed

- **`compiler.normalize` — silent token corruption** (`sin(ln(x))` → `silog(log(x))`):
  bare-string `.replace("ln", "log")` and `.replace("sen", "sin")` corrupted any
  identifier that contained the sub-sequence `ln` or `sen` (e.g. `sin`, `plan`).
  Both replacements now use word-boundary regexes (`\bln\b`, `\bsen\b`) so only
  the standalone tokens are rewritten. `"ln"` removed from `ALLOWED_FUNCTIONS`
  (it was always normalised to `"log"` before the lexer ran; keeping it there was
  misleading).
- **`_downsample` — first / last elements not preserved**: the index arithmetic
  `values[int(i * step)]` could skip `values[0]` and `values[-1]`, truncating
  plots and time-series at both ends. The sampler now always includes the boundary
  indices and adjusts the middle points to stay within `max_points`.
- **`integral_simpson13` — opaque error for odd `n`**: the previous message
  (`"n must be even for Simpson 1/3"`) gave no remediation hint to the calling
  LLM. The new message includes the suggested correction, e.g.
  `"Try n=4 instead."`.

### Added (CI / release)

- `.github/workflows/release.yml`: pushing a `v*` tag runs the full CI suite and
  then publishes to PyPI and npm automatically. Both registries use GitHub OIDC
  (Trusted Publishing), so no API tokens or repository secrets are stored. The
  workflow verifies the tag matches the version in `pyproject.toml` and
  `package.json`, builds the distributions in a credential-free job, and npm
  publishes with an automatic provenance attestation.
- `ci.yml` is now reusable (`workflow_call`) so the release workflow can gate
  publishing on a green test run.

### Changed

- CI actions updated to `actions/checkout@v6`, clearing the Node.js 20
  deprecation warning.

## [0.2.0] - 2026-10-16

### Added

- **Visual output**: every tool accepts `plot: bool = False`. With `plot=True`
  the response keeps the JSON result and adds a rendered PNG as an MCP image
  content block (phase portraits, solution curves, bifurcation diagrams,
  Monte Carlo convergence and distributions, integration areas, root brackets,
  Lagrange polynomial, finite-difference tangent, ...).
- `mathmethods/plots.py`: plot registry + lazy Matplotlib rendering
  (`Figure` + `FigureCanvasAgg`, deterministic PNG bytes).
- Optional dependency extra `viz` (`matplotlib>=3.8`). The default install is
  unchanged and the stdio startup stays light; using `plot=True` without the
  extra returns an actionable error.
- `tests/test_plots.py`: registry/tool coverage, `plot=False` contract
  regression, deterministic rendering, missing-Matplotlib error path, and an
  end-to-end MCP protocol test asserting `image/png` content.
- CI now runs two jobs: the default install (skips rendering tests) and one with
  `--extra viz` (full plotting suite).

### Changed

- All tools are registered with `structured_output=False` so returning
  `[dict, Image]` is well-defined regardless of type annotations.
- Server `instructions` mention the `plot` option.

### Fixed

- `plot=True` no longer fails on scalar-valued SymPy functions (`gamma`,
  `loggamma`, `erf`, `factorial`, `binomial`, ...): the plot sampler falls back
  to per-point evaluation.
- `_trim_large_arrays` now also downsamples long numeric lists that contain
  `None` (NaN) entries.
- 2D nonlinear phase portraits now overlay the nullclines from the mesh grid.

## [0.1.0] - 2026-10-15

### Added

- Initial release: 29 numerical-methods tools (roots, integration,
  differentiation, ODE, interpolation, Monte Carlo, dynamic systems) over MCP,
  hardened expression parsing, PyPI + npm packaging and CI.

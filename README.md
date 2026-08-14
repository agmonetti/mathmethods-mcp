# MathematicalMethods MCP Server

[![MCP Server](https://badge.mcpx.dev?type=server)](https://modelcontextprotocol.io/introduction)
[![License](https://img.shields.io/github/license/agmonetti/mathmethods-mcp)](LICENSE)
[![CI](https://github.com/agmonetti/mathmethods-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/agmonetti/mathmethods-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mathmethods-mcp)](https://pypi.org/project/mathmethods-mcp/)

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes a
numerical-methods core as tools an LLM agent can call directly from a chat
(VS Code, Zed, Claude, opencode, etc.). It is built on top of the
numerical-methods engine of the academic project `modeladoYsimulacion-web`
(UADE); the math core is **vendored** into this repository so the server is
fully self-contained.

## Quick start

```bash
claude mcp add mathmethods-mcp -- uvx mathmethods-mcp
```

Any MCP client registers the server with the same one-liner command —
`uvx mathmethods-mcp` (a Python package that needs no cloning, venv or paths):

```json
{ "command": "uvx", "args": ["mathmethods"] }
```

<details>
<summary>Run from a checkout instead (for development)</summary>

```bash
git clone https://github.com/agmonetti/mathmethods-mcp.git
cd mathmethods
uv sync --extra dev
uv run mathmethods
```

Every client config below also works with
`uv run --frozen --project <checkout> python <checkout>/server.py` in place of
`uvx mathmethods-mcp`.
</details>

## Tools

### Root finding

| Tool                    | What it does                                        |
| ----------------------- | --------------------------------------------------- |
| `root_bisection`        | Bisection on `[a, b]` (requires a sign change)      |
| `root_newton_raphson`   | Newton–Raphson with numeric derivative              |
| `root_punto_fijo`       | Fixed-point iteration `x = g(x)`                    |
| `root_aitken`           | Aitken Δ² acceleration of fixed point               |
| `root_comparar`         | All four methods compared on the same problem       |

### Numerical integration

| Tool                    | What it does                                        |
| ----------------------- | --------------------------------------------------- |
| `integral_rectangulo`   | Composite midpoint rule                             |
| `integral_trapecio`     | Composite trapezoidal rule                          |
| `integral_simpson13`    | Composite Simpson 1/3 (n even)                      |
| `integral_simpson38`    | Composite Simpson 3/8 (n multiple of 3)             |
| `integral_comparar`     | All four rules compared on the same integral        |

### Differentiation

| Tool                    | What it does                                        |
| ----------------------- | --------------------------------------------------- |
| `finite_differences`    | Forward/backward/central 1st & 2nd derivatives      |

### ODE and interpolation

| Tool                      | What it does                                            |
| ------------------------- | ------------------------------------------------------- |
| `ode_rk4`                 | Runge–Kutta 4 (4th order)                               |
| `ode_heun`                | Heun predictor–corrector (2nd order)                    |
| `ode_euler`               | Explicit Euler (1st order)                              |
| `interpolation_lagrange`  | Lagrange interpolating polynomial                       |

### Monte Carlo

| Tool                    | What it does                                            |
| ----------------------- | ------------------------------------------------------- |
| `mc_hit_or_miss_1d`     | Hit-or-miss estimator (correct for sign-changing f)     |
| `mc_valor_promedio_1d`  | Mean-value estimate of `∫ₐᵇ f(x) dx`                    |
| `mc_valor_promedio_2d`  | Mean-value estimate of a double integral                |
| `mc_valor_promedio_3d`  | Mean-value estimate of a triple integral                |
| `mc_estadistico_1d`     | M×N replicated experiment with statistical analysis     |
| `mc_convergencia_1d`    | Running average showing the estimate converging         |

### Dynamic systems

| Tool                          | What it does                                              |
| ----------------------------- | --------------------------------------------------------- |
| `dynamic_1d_solve`            | Equilibria, stability, phase portrait and time series     |
| `dynamic_1d_equilibria`       | Find and classify the equilibria of `x' = f(x)`           |
| `dynamic_1d_bifurcation`      | Equilibria vs parameter (bifurcation diagram)             |
| `dynamic_2d_linear_solve`     | Linear `X' = A·X + B`: classification, eigenvalues, analytic solution |
| `dynamic_2d_nonlinear_solve`  | Nonlinear `x' = f(x,y)`: equilibria, Jacobian, nullclines |
| `dynamic_2d_conservative_solve` | Divergence-free check, Hamiltonian/energy, closed orbits |
| `dynamic_2d_lanchester_solve` | Lanchester combat model with analytic time-to-annihilation |
| `dynamic_2d_nonhomogeneous_solve` | Non-homogeneous `X' = A·X + B(t)` with time-varying forcing |

Math expressions use Python/SymPy syntax: `x**2`, `sin(x)`, `exp(x)`,
`sqrt(x)`, `log(x)`. Common shorthand is accepted too: `e^x`, `sen(x)`, `ln(x)`
and the caret `^` for powers. The Greek combat parameters of Lanchester use the
Unicode symbols `α β γ ε μ δ`.

## Project layout

```
modelo-mat-mcp/
├── server.py                 # FastMCP app + all tools
├── mathmethods/
│   ├── compiler.py           # hardened expression validation (whitelist, caps)
│   ├── server.py             # FastMCP app and tool definitions
│   └── core/                 # vendored math core (from modeladoYsimulacion-web)
│       ├── root_finding.py   ├── integration.py
│       ├── ode.py            ├── interpolation.py
│       ├── differentiation.py├── monte_carlo.py
│       ├── dynamic_1d.py     ├── dynamic_2d_linear.py
│       ├── dynamic_2d_non_homogeneous.py ├── dynamic_2d_nonlinear.py
│       ├── dynamic_2d_conservative.py ├── dynamic_2d_lanchester.py
│       └── utils.py
├── tests/                    # test_tools.py + test_dynamic_tools.py
├── mcp.example.json          # server registration template (copy to .vscode/mcp.json)
├── requirements.txt
└── pyproject.toml
```

## Install

```bash
cd modelo-mat-mcp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> After creating the venv, verify it is isolated (`venv/bin/python -c "import sys; print(sys.prefix)"`
> should print the venv path, not `/usr`). If your system Python produces a broken venv, try
> `python3 -m venv --copies venv`.

## Run

**Local (STDIO)** — default transport, used by VS Code / Claude Desktop:

```bash
uv run python server.py
```

**Remote (Streamable HTTP)** — the server prints a URL such as `http://127.0.0.1:8000/mcp`:

```bash
MCP_TRANSPORT=streamable-http uv run python server.py
```

The transport can also be chosen with the `MCP_TRANSPORT` environment variable
(`stdio` | `streamable-http` | `sse`), and the HTTP host/port with
`MCP_HTTP_HOST` / `MCP_HTTP_PORT` (defaults `127.0.0.1:8000`).

## Connect from a client

Every client registers the **same command**, `uvx mathmethods-mcp` (no paths, no
venv). If the server is not published yet or you work from a checkout, use
`uv run --frozen --project <PROJ> python <PROJ>/server.py` instead.

**Remote (Streamable HTTP)** — optional; start it once in a terminal, then
point the client at `http://127.0.0.1:8000/mcp`:

```bash
MCP_TRANSPORT=streamable-http uvx mathmethods-mcp
```

<details>
<summary><b>VS Code</b></summary>

Create `.vscode/mcp.json` (git-ignored) — or copy `mcp.example.json`:

```json
{
  "servers": {
    "modelo-mat-stdio": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mathmethods"]
    },
    "modelo-mat-http": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Open the file and press **Start** next to the server you want; reload the
window if it doesn't appear (`Developer: Reload Window`).

</details>

<details>
<summary><b>Zed</b></summary>

Add the entry under `context_servers` (note: **not** `mcp_servers`) in
`~/.config/zed/settings.json` or the project-level `.zed/settings.json`:

```json
{
  "context_servers": {
    "modelo-mat": {
      "command": "uvx",
      "args": ["mathmethods"]
    }
  }
}
```

You can also manage them via **Settings → AI → MCP Servers**.

</details>

<details>
<summary><b>opencode / OpenChamber</b></summary>

Both opencode and the OpenChamber desktop app share the same configuration
format. Add the entry under `mcp` in `opencode.json` (project root) or in the
global `~/.config/opencode/opencode.jsonc`:

```json
{
  "mcp": {
    "modelo-mat": {
      "type": "local",
      "command": ["uvx", "mathmethods"],
      "enabled": true
    }
  }
}
```

Or register it with the CLI (equivalent):

```bash
opencode mcp add modelo-mat -- uvx mathmethods-mcp
```

For a remote server running on `http://127.0.0.1:8000/mcp`:

```json
{
  "mcp": {
    "modelo-mat": {
      "type": "remote",
      "url": "http://127.0.0.1:8000/mcp",
      "enabled": true
    }
  }
}
```

Verify with `opencode mcp list`.

</details>

<details>
<summary><b>Antigravity</b></summary>

Add the entry under `mcpServers` in the Antigravity config file, typically
`~/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "modelo-mat": {
      "command": "uvx",
      "args": ["mathmethods"]
    }
  }
}
```

If the file path differs on your install, use the in-IDE **Settings →
Integrations → MCP Servers** panel instead, which writes the same format.

</details>

<details>
<summary><b>GitHub Copilot CLI</b></summary>

The GitHub Copilot CLI (`copilot`) lets you add a server interactively:

```bash
copilot
```

then inside the session:

```text
/mcp add
  Server name:  modelo-mat
  Server type:  1 (Local/STDIO)
  Command:      uvx mathmethods-mcp
```

Press `Ctrl+S` to save. The settings are stored in
`~/.copilot/mcp-config.json` (top-level `mcpServers`); check the connection
with `/mcp show`.

</details>

<details>
<summary><b>Claude Desktop / Claude Code</b></summary>

Both use the `mcpServers` format. In Claude Desktop, edit
`claude_desktop_config.json`; in Claude Code:

```bash
claude mcp add mathmethods-mcp -- uvx mathmethods-mcp
```

```json
{
  "mcpServers": {
    "modelo-mat": {
      "command": "uvx",
      "args": ["mathmethods"]
    }
  }
}
```
</details>

### Verify with the MCP Inspector

```bash
npx @modelcontextprotocol/inspector node server.py   # or
npx @modelcontextprotocol/inspector --transport http http://127.0.0.1:8000/mcp
```

## Example usage

Ask your agent things like:

- "Find the root of `x^3 - 3x + 1` in `[0, 1]`."
- "Integrate `sin(x)/x` from 0 to 1 using Simpson with n=10."
- "Solve `y' = y` with y(0)=1 from x=0 to x=1 with step 0.1 (RK4)."
- "Build the Lagrange polynomial through (0,1), (1,3), (2,7) and evaluate at 1.5."
- "Estimate the integral of `sin(x)` over `[0, 2pi]` with Monte Carlo hit-or-miss."
- "Find the equilibria of the logistic model `x' = mu*x*(1 - x/K)` with K=2, mu=1."
- "Classify the 2D system `x' = 2x - y`, `y' = x + 2y` and sketch its trajectories."
- "Simulate a Lanchester battle x'=-αy, y'=-βx with α=1, β=2, 100 vs 80 soldiers."

## Security

The server is **read-only**: the tools only compute numbers, they never touch
the filesystem, the network or any destructive operation. Still, the inputs are
driven by an LLM, so defense in depth is applied:

- **Expression hardening** (`mathmethods/compiler.py` + `mathmethods/core/utils.py`):
  length cap, symbol whitelist, function whitelist, and a lexical gate that
  rejects attribute access (`.`/`__`) and unknown tokens BEFORE SymPy parses.
  SymPy's `sympify`/`parse_expr` can execute arbitrary Python (verified RCE),
  so every parse site — in this project and in the upstream backend — routes
  through the gate.
- **Input caps**: iteration/subinterval/step/point counts are bounded to avoid
  pathological CPU/RAM usage.
- **Exact tool descriptions**: the LLM picks tools by their metadata, so
  descriptions stay accurate (guards against tool-poisoning attacks).
- **Prompt injection**: even if the model is tricked, the worst it can do is ask
  for another computation. There are no privileged side channels.

## Known limitations

- The vendored core is inherited from the upstream project and kept as-is
  (Spanish identifiers, etc.).
- `dynamic_2d_nonhomogeneous_solve` with **time-varying forcing on a
  non-diagonal matrix A** shows the homogeneous solution only (the particular
  term is computed for diagonal systems); the numeric RK4 trajectory is always
  correct.
- The 1D bifurcation table is downsampled to 300 rows for readability.

## Publishing to PyPI

The package is publish-ready (`uv build` succeeds and the wheel exposes all
tools). To release:

```bash
uv build
uv publish          # requires a PyPI token: `uv login` or UV_PUBLISH_TOKEN
```

Once published, every client config just works with `uvx mathmethods-mcp` (no
paths, no venv). Bump `version` in `pyproject.toml` before each release.

## Keeping the vendored core in sync

The math lives in `modeladoYsimulacion-web/backend/app/methods/`. When the
upstream code changes, copy the files here again:

```bash
cp ../modeladoYsimulacion-web/backend/app/methods/{root_finding,integration,ode,interpolation,monte_carlo,dynamic_1d,dynamic_2d_linear,dynamic_2d_non_homogeneous,dynamic_2d_nonlinear,dynamic_2d_conservative,dynamic_2d_lanchester}.py mathmethods/core/
cp ../modeladoYsimulacion-web/backend/app/core/utils.py mathmethods/core/utils.py
```

Then rewrite the `from app.core.utils import ...` imports to `from .utils
import ...` in the copied files.

## Test

```bash
uv run pytest
```

## Roadmap

- Translate the vendored core to English (manual, when time allows).
- Server-side CI is wired up (`.github/workflows/ci.yml`); coverage report next.
- Optional MCP resources/prompts (e.g. a theorem reference) on top of the tools.

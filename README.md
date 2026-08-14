# MathematicalMethods MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes a
numerical-methods core as tools an LLM agent can call directly from a chat
(VS Code, Claude Desktop, etc.).

Built on top of the numerical-methods engine of the academic project
`modeladoYsimulacion-web` (UADE). The math core is **vendored** into this
repository so the server is fully self-contained and publishable.

## Tools

### Root finding, integration, ODE and interpolation

| Tool                  | What it does                                          | Math method      |
| --------------------- | ----------------------------------------------------- | ---------------- |
| `root_bisection`      | Find a root of `f(x) = 0` in `[a, b]`                 | Bisection        |
| `integral_simpson13`  | Approximate `∫ₐᵇ f(x) dx` with composite Simpson 1/3  | Newton–Cotes     |
| `ode_rk4`             | Solve `y' = f(x, y)`, `y(x0) = y0` up to `xf`         | Runge–Kutta 4    |
| `interpolation_lagrange` | Build the Lagrange polynomial through given points  | Lagrange         |

### Monte Carlo

| Tool                    | What it does                                            |
| ----------------------- | ------------------------------------------------------- |
| `mc_hit_or_miss_1d`     | Hit-or-miss estimator (correct for sign-changing f)     |
| `mc_valor_promedio_1d`  | Mean-value estimate of `∫ₐᵇ f(x) dx`                    |
| `mc_valor_promedio_2d`  | Mean-value estimate of a double integral                |
| `mc_valor_promedio_3d`  | Mean-value estimate of a triple integral                |
| `mc_estadistico_1d`     | M×N replicated experiment with statistical analysis     |

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
│       ├── monte_carlo.py    ├── dynamic_1d.py
│       ├── dynamic_2d_linear.py ├── dynamic_2d_non_homogeneous.py
│       ├── dynamic_2d_nonlinear.py ├── dynamic_2d_conservative.py
│       └── dynamic_2d_lanchester.py └── utils.py
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

Every client below registers the same server. Replace the placeholders:

- `<UV>` — path to the `uv` binary (or just `uv` if it is on your `PATH`)
- `<PROJ>` — absolute path to this checkout, e.g. `/home/me/modelo-mat-mcp`

**Local (STDIO)** — the client launches the server for you; only needs `uv`:

```
<UV> run --frozen --project <PROJ> python <PROJ>/server.py
```

**Remote (Streamable HTTP)** — start it once in a terminal, then point the
client at `http://127.0.0.1:8000/mcp`:

```bash
MCP_TRANSPORT=streamable-http uv run python server.py
```

<details>
<summary><b>VS Code</b></summary>

Your `.vscode/mcp.json` is **machine-specific and git-ignored** (it contains
local paths). Copy the template and adjust the paths:

```bash
cp mcp.example.json .vscode/mcp.json
```

```json
{
  "servers": {
    "modelo-mat-stdio": {
      "type": "stdio",
      "command": "<UV>",
      "args": [
        "run", "--frozen", "--project", "<PROJ>",
        "python", "<PROJ>/server.py"
      ]
    },
    "modelo-mat-http": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Open `.vscode/mcp.json` and press **Start** next to the server you want. The
STDIO entry is started on demand by VS Code; the HTTP entry needs the server
already running (see above). Reload the window after editing
(`Developer: Reload Window`) if the servers don't appear.

</details>

<details>
<summary><b>Zed</b></summary>

Add the entry under `context_servers` (note: **not** `mcp_servers`) in
`~/.config/zed/settings.json` or the project-level `.zed/settings.json`:

```json
{
  "context_servers": {
    "modelo-mat": {
      "command": "<UV>",
      "args": [
        "run", "--frozen", "--project", "<PROJ>",
        "python", "<PROJ>/server.py"
      ]
    }
  }
}
```

You can also manage them via **Settings → AI → MCP Servers**. For the HTTP
transport Zed connects natively over STDIO, so the local entry above is the
recommended way.

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
      "command": ["<UV>", "run", "--frozen", "--project", "<PROJ>", "python", "<PROJ>/server.py"],
      "enabled": true
    }
  }
}
```

Or register it with the CLI (equivalent):

```bash
opencode mcp add modelo-mat -- <UV> run --frozen --project <PROJ> python <PROJ>/server.py
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
      "command": "<UV>",
      "args": [
        "run", "--frozen", "--project", "<PROJ>",
        "python", "<PROJ>/server.py"
      ]
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
  Command:      <UV> run --frozen --project <PROJ> python <PROJ>/server.py
```

Press `Ctrl+S` to save. The settings are stored in
`~/.copilot/mcp-config.json` (top-level `mcpServers`); check the connection
with `/mcp show`.

</details>

<details>
<summary><b>Claude Desktop / Claude Code</b></summary>

Both use the `mcpServers` format. In Claude Desktop, edit
`claude_desktop_config.json`; in Claude Code use `claude mcp add`:

```json
{
  "mcpServers": {
    "modelo-mat": {
      "command": "<UV>",
      "args": [
        "run", "--frozen", "--project", "<PROJ>",
        "python", "<PROJ>/server.py"
      ]
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

# MathematicalMethods MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes a
numerical-methods core as tools an LLM agent can call directly from a chat
(VS Code, Claude Desktop, etc.).

Built on top of the numerical-methods engine of the academic project
`modeladoYsimulacion-web` (UADE). The math core is **vendored** into this
repository so the server is fully self-contained and publishable.

## Tools

| Tool                  | What it does                                          | Math method      |
| --------------------- | ----------------------------------------------------- | ---------------- |
| `root_bisection`      | Find a root of `f(x) = 0` in `[a, b]`                 | Bisection        |
| `integral_simpson13`  | Approximate `∫ₐᵇ f(x) dx` with composite Simpson 1/3  | Newton–Cotes     |
| `ode_rk4`             | Solve `y' = f(x, y)`, `y(x0) = y0` up to `xf`         | Runge–Kutta 4    |
| `interpolation_lagrange` | Build the Lagrange polynomial through given points  | Lagrange         |

Math expressions use Python/SymPy syntax: `x**2`, `sin(x)`, `exp(x)`,
`sqrt(x)`, `log(x)`. Common shorthand is accepted too: `e^x`, `sen(x)`, `ln(x)`
and the caret `^` for powers.

## Project layout

```
modelo-mat-mcp/
├── server.py                 # FastMCP app + the 4 tools
├── mathmethods/
│   ├── compiler.py           # hardened expression validation (whitelist, caps)
│   └── core/                 # vendored math core (from modeladoYsimulacion-web)
│       ├── root_finding.py
│       ├── integration.py
│       ├── ode.py
│       └── interpolation.py
├── tests/test_tools.py
├── mcp.example.json            # server registration template (copy to .vscode/mcp.json)
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

### VS Code

Your `.vscode/mcp.json` is **machine-specific and git-ignored** (it contains
local paths). Copy the template and adjust the paths to your checkout:

```bash
cp mcp.example.json .vscode/mcp.json
```

The STDIO entry launches the server through `uv` (using the project's
`.venv`/`uv.lock`); if you prefer a classic virtualenv, use `venv/bin/python`
instead of `uv run`.

```json
{
  "servers": {
    "modelo-mat-stdio": {
      "type": "stdio",
      "command": "/absolute/path/to/uv",
      "args": [
        "run", "--frozen", "--project", "/absolute/path/to/modelo-mat-mcp",
        "python", "/absolute/path/to/modelo-mat-mcp/server.py"
      ]
    },
    "modelo-mat-http": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

For the HTTP entry, start the server first in a terminal:

```bash
MCP_TRANSPORT=streamable-http uv run python server.py
```

With the STDIO entry you only need the server **running** for the HTTP entry;
STDIO is started by the client on demand.

### Claude Desktop

Add a `mcpServers` entry pointing at the same `command`/`args`.

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

## Security

The server is **read-only**: the tools only compute numbers, they never touch
the filesystem, the network or any destructive operation. Still, the inputs are
driven by an LLM, so defense in depth is applied:

- **Expression hardening** (`mathmethods/compiler.py`): length cap, symbol
  whitelist, function whitelist, and `sympify` (no `eval`/`exec`, so RCE
  payloads are rejected by the parser).
- **Input caps**: iteration/subinterval/step/point counts are bounded to avoid
  pathological CPU/RAM usage.
- **Exact tool descriptions**: the LLM picks tools by their metadata, so
  descriptions stay accurate (guards against tool-poisoning attacks).
- **Prompt injection**: even if the model is tricked, the worst it can do is ask
  for another computation. There are no privileged side channels.

## Known limitations

- The vendored core is inherited from the upstream project and kept as-is
  (Spanish identifiers, etc.). Only MVP-relevant bugs were patched: `e^x`
  parsing, the ODE exact-solver fallback and endpoint handling.
- Monte Carlo and 1D/2D dynamic-system tools are not exposed yet (roadmap).

## Keeping the vendored core in sync

The math lives in `modeladoYsimulacion-web/backend/app/methods/`. When the
upstream code changes, copy the four files here again:

```bash
cp ../modeladoYsimulacion-web/backend/app/methods/{root_finding,integration,ode,interpolation}.py \
   mathmethods/core/
```

Mind the local bug fixes (search for `re.sub(r'(?<![A-Za-z0-9_])[eE]\^'` and the
`f_exacta` fallback in `ode.py`) so they are not lost.

## Test

```bash
source venv/bin/activate
pytest
```

## Roadmap

- Monte Carlo tools (hit-or-miss with signed integrands, mean value 1D/2D/3D).
- Dynamic systems: 1D equilibria/bifurcations, 2D linear / non-linear /
  conservative / Lanchester.
- Translate the vendored core to English (manual, when time allows).

"""MathematicalMethods MCP server entry point.

Run locally (STDIO, default for VS Code / Claude Desktop):

    mcp run server.py

Run over HTTP (Streamable HTTP, like an API):

    mcp run server.py --transport streamable-http --port 8000
"""

from mathmethods.server import (
    mcp,  # noqa: F401  (re-exported for `mcp run`)
    run,
)

if __name__ == "__main__":
    run()

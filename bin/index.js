#!/usr/bin/env node

"use strict";

const { spawn, execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

function commandExists(cmd) {
  try {
    const isWindows = process.platform === "win32";
    const checkCmd = isWindows ? `where ${cmd}` : `command -v ${cmd}`;
    execSync(checkCmd, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function canImportMathmethods(pythonCmd) {
  try {
    execSync(`${pythonCmd} -c "import mathmethods"`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function run() {
  const args = process.argv.slice(2);

  // Check if running directly inside a local development checkout
  const repoRoot = path.resolve(__dirname, "..");
  const isLocalDev =
    fs.existsSync(path.join(repoRoot, "mathmethods", "core")) &&
    fs.existsSync(path.join(repoRoot, "server.py"));

  const venvPython =
    process.platform === "win32"
      ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
      : path.join(repoRoot, ".venv", "bin", "python");

  let executable;
  let execArgs;

  if (isLocalDev && fs.existsSync(venvPython)) {
    executable = venvPython;
    execArgs = [path.join(repoRoot, "server.py"), ...args];
  } else if (commandExists("uvx")) {
    executable = "uvx";
    execArgs = ["mathmethods-mcp", ...args];
  } else if (commandExists("pipx")) {
    executable = "pipx";
    execArgs = ["run", "mathmethods-mcp", ...args];
  } else if (commandExists("python3") && canImportMathmethods("python3")) {
    executable = "python3";
    execArgs = ["-m", "mathmethods.server", ...args];
  } else if (commandExists("python") && canImportMathmethods("python")) {
    executable = "python";
    execArgs = ["-m", "mathmethods.server", ...args];
  } else {
    process.stderr.write(
      "\n[mathmethods-mcp] Error: No suitable Python execution environment found.\n" +
        "[mathmethods-mcp] mathmethods-mcp is powered by Python numerical libraries and recommends 'uv'.\n\n" +
        "[mathmethods-mcp] Quick install uv (fastest & isolated):\n" +
        "  curl -LsSf https://astral.sh/uv/install.sh | sh     (macOS / Linux)\n" +
        "  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\" (Windows)\n\n" +
        "Alternatively, install mathmethods-mcp with pip:\n" +
        "  pip install mathmethods-mcp\n\n"
    );
    process.exit(1);
  }

  const child = spawn(executable, execArgs, {
    stdio: "inherit",
    shell: false,
    env: {
      ...process.env,
      PYTHONWARNINGS: process.env.PYTHONWARNINGS || "ignore",
    },
  });

  child.on("error", (err) => {
    process.stderr.write(`[mathmethods-mcp] Failed to start server: ${err.message}\n`);
    process.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 0);
    }
  });

  process.on("SIGINT", () => child.kill("SIGINT"));
  process.on("SIGTERM", () => child.kill("SIGTERM"));
}

run();

#!/usr/bin/env bash

# ==============================================================================
# Open Audio Studio Bootstrapper & Running Script
# ==============================================================================
# Automatically manages:
#   1. Virtual environment creation & activation
#   2. Python dependency installation in editable mode
#   3. Next.js package resolution using pnpm (falling back to npx pnpm)
#   4. Concurrent server orchestration with graceful SIGINT/SIGTERM trapping
# ==============================================================================

set -euo pipefail

# ANSI color codes for rich console formatting
INFO='\033[1;36m'   # Cyan
SUCCESS='\033[1;32m' # Green
WARN='\033[1;33m'    # Yellow
ERROR='\033[1;31m'   # Red
RESET='\033[0m'      # Text Reset

echo -e "${INFO}====================================================================${RESET}"
echo -e "${INFO}                 🎙️  WELCOME TO OPEN AUDIO STUDIO  🎙️                 ${RESET}"
echo -e "${INFO}====================================================================${RESET}"

# Step 1: Verify Python Installation
if ! command -v python3 &>/dev/null; then
    echo -e "${ERROR}Error: python3 is not installed on your system. Please install it first.${RESET}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${SUCCESS}✓ Python 3 discovered: version ${PYTHON_VERSION}${RESET}"

# Step 2: Establish Python Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${WARN}• Virtual environment not found. Generating a new one at .venv...${RESET}"
    python3 -m venv "$VENV_DIR"
    echo -e "${SUCCESS}✓ Virtual environment created successfully.${RESET}"
fi

echo -e "${INFO}• Activating virtual environment...${RESET}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Step 3: Install/Update Python Modules
echo -e "${INFO}• Installing core packages, SDK, and FastAPI server in editable mode...${RESET}"
python -m pip install --upgrade pip
pip install -e packages/core -e packages/sdk -e apps/server
pip install pytest pytest-asyncio ruff mypy httpx soundfile librosa numpy torch peft transformers bitsandbytes
echo -e "${SUCCESS}✓ Python package dependencies successfully synced.${RESET}"

# Step 4: Resolve Node.js and package manager dependencies
if ! command -v node &>/dev/null; then
    echo -e "${ERROR}Error: Node.js is not installed. Please install Node.js 18+ to run the web interface.${RESET}"
    exit 1
fi
echo -e "${SUCCESS}✓ Node.js discovered: version $(node -v)${RESET}"

# Choose package manager
PM_CMD="pnpm"
if ! command -v pnpm &>/dev/null; then
    echo -e "${WARN}• Global pnpm not detected. Utilizing 'npx pnpm' virtual wrapper...${RESET}"
    PM_CMD="npx pnpm"
fi

echo -e "${INFO}• Installing Next.js frontend dependencies...${RESET}"
cd apps/web
$PM_CMD install
cd - >/dev/null

echo -e "${SUCCESS}✓ Frontend web package dependencies successfully synced.${RESET}"

# Step 5: Check and free up binding ports if occupied
check_port() {
    local port=$1
    if command -v lsof &>/dev/null; then
        lsof -i :"$port" -t | xargs kill -9 2>/dev/null || true
    elif command -v fuser &>/dev/null; then
        fuser -k "$port"/tcp &>/dev/null || true
    fi
}

echo -e "${INFO}• Releasing occupied ports...${RESET}"
check_port 8000 # Backend API port
check_port 3000 # Next.js frontend port

# Step 6: Concurrently Boot both Services
echo -e "${SUCCESS}✓ Ready! Booting Open Audio Studio stack...${RESET}"
echo -e "${INFO}--------------------------------------------------------------------${RESET}"
echo -e "${SUCCESS}  FastAPI API Backend  : http://localhost:8000${RESET}"
echo -e "${SUCCESS}  Next.js Web Frontend : http://localhost:3000${RESET}"
echo -e "${INFO}--------------------------------------------------------------------${RESET}"
echo -e "${WARN}Press Ctrl+C to terminate both servers.${RESET}"
echo -e "${INFO}--------------------------------------------------------------------${RESET}"

# Trap termination signals to gracefully clean up background subprocesses
cleanup() {
    echo -e "\n${WARN}• Terminating background servers...${RESET}"
    kill 0
    echo -e "${SUCCESS}✓ Servers successfully stopped. Goodbye!${RESET}"
}
trap cleanup SIGINT SIGTERM

# Launch Backend
cd apps/server
uvicorn oas_server.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd - >/dev/null

# Launch Frontend
cd apps/web
$PM_CMD dev &
FRONTEND_PID=$!
cd - >/dev/null

# Keep script alive waiting for child processes
wait "$BACKEND_PID" "$FRONTEND_PID"

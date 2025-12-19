#!/bin/bash
# start.sh
# Deterministic boot sequence for Environment Mastery Engine (EME)
# This script turns the key; it does not drive the car.

set -euo pipefail

# Exit immediately on error, undefined variables, and pipe failures
# Clear error output on failure
exec 2>&1

echo "=== EME Boot Sequence ==="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo

# ============================================================
# Phase 1: Verify execution context
# ============================================================
echo "--- Phase 1: Verifying execution context ---"

# Verify Linux platform
OS_NAME=$(uname -s)
if [ "$OS_NAME" != "Linux" ]; then
    echo "ERROR: Unsupported platform: $OS_NAME"
    echo "EME requires Linux for actuation"
    exit 1
fi
echo "✓ Platform: Linux"

# Verify Python 3.9+ availability
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python: $PYTHON_VERSION"

# Verify working directory contains project structure
if [ ! -d "bootstrap" ] || [ ! -d "core" ] || [ ! -d "interfaces" ] || [ ! -d "execution" ]; then
    echo "ERROR: Incorrect working directory"
    echo "Must run from project root containing bootstrap/, core/, interfaces/, execution/"
    exit 1
fi
echo "✓ Project structure verified"

# ============================================================
# Phase 2: Load environment
# ============================================================
echo
echo "--- Phase 2: Loading environment ---"

# Export non-secret environment variables only
# These are mechanical settings, not secrets
export PYTHONPATH="$PWD:$PYTHONPATH"
export EME_ROOT="$PWD"
export EME_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")

# Display environment summary
echo "✓ PYTHONPATH set"
echo "✓ EME_ROOT: $EME_ROOT"
echo "✓ EME_TIMESTAMP: $EME_TIMESTAMP"

# Note: Secrets must be set in environment before running this script
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "WARNING: DEEPSEEK_API_KEY not set in environment"
    echo "Note: This will cause bootstrap/deps_installer.py to fail"
fi

# Display DISPLAY variable for X11/Wayland (mechanical only)
if [ -n "${DISPLAY:-}" ]; then
    echo "✓ DISPLAY: $DISPLAY"
else
    echo "WARNING: DISPLAY environment variable not set"
fi

# ============================================================
# Phase 3: Run bootstrap sequence (strict order)
# ============================================================
echo
echo "--- Phase 3: Running bootstrap sequence ---"
echo "Note: Bootstrap steps run in strict order, no retries"

# Step 3.1: System probe
echo "--- 3.1: Running system_probe.py ---"
$PYTHON_CMD -m bootstrap.system_probe
echo "✓ System probe completed"

# Step 3.2: Vision installer
echo "--- 3.2: Running vision_installer.py ---"
$PYTHON_CMD -m bootstrap.vision_installer
echo "✓ Vision installer completed"

# Step 3.3: Body installer
echo "--- 3.3: Running body_installer.py ---"
$PYTHON_CMD -m bootstrap.body_installer
echo "✓ Body installer completed"

# Step 3.4: Dependencies installer
echo "--- 3.4: Running deps_installer.py ---"
$PYTHON_CMD -m bootstrap.deps_installer
echo "✓ Dependencies installer completed"

# Step 3.5: Sanity checks
echo "--- 3.5: Running sanity_checks.py ---"
$PYTHON_CMD -m bootstrap.sanity_checks
echo "✓ Sanity checks completed"

# ============================================================
# Phase 4: Validate configuration
# ============================================================
echo
echo "--- Phase 4: Validating configuration ---"

CONFIG_FILE="config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_SIZE=$(wc -c < "$CONFIG_FILE")
if [ "$CONFIG_SIZE" -eq 0 ]; then
    echo "ERROR: Configuration file is empty: $CONFIG_FILE"
    exit 1
fi

echo "✓ Configuration file: $CONFIG_FILE ($CONFIG_SIZE bytes)"

# ============================================================
# Phase 5: Start the organism
# ============================================================
echo
echo "--- Phase 5: Starting organism ---"
echo "Handing control to life loop"
echo "Config: $CONFIG_FILE"
echo

# Execute life loop
# Note: No wrapping, no supervision, no error catching
# The loop runs until termination or failure
exec $PYTHON_CMD -m core.life_loop --config "$CONFIG_FILE"

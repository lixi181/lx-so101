#!/bin/sh
set -eu
workspace_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$workspace_dir"
exec .venv/bin/python -m so_ws.scripts.verify_model

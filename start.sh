#!/usr/bin/env bash
set -euo pipefail
exec streamlit run code/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true
#!/usr/bin/env bash
set -euo pipefail
pip install -r requirements.txt
python -m code.ingestion
python -m code.ingestion --chunk
python -m code.ingestion --index
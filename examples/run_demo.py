#!/usr/bin/env python3
"""
Standalone demo: runs the full pipeline (indexing, both retrieval
directions, evaluation, failure analysis, hard-negative reranking, HTML
report) on the 50-item synthetic mock dataset. No downloads, no GPU,
completes in well under a minute.

Run with:
    python examples/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import main as run_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_main(["--mode", "demo", "--output", "examples/demo_report.html"]))

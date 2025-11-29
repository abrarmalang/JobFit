#!/usr/bin/env python
"""
Run Data Collection Pipeline

Cross-platform script to fetch jobs from Adzuna API.

Usage:
    python scripts/run_collection_adzuna.py
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from workers.collection.main_collection import main

if __name__ == "__main__":
    main()

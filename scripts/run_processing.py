#!/usr/bin/env python
"""
Run Data Processing Pipeline

Cross-platform script to consolidate raw data into processed dataset.

Usage:
    python scripts/run_processing.py
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from workers.processing.main_processing import main

if __name__ == "__main__":
    main()

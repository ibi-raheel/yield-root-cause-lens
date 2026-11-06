import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
SECOM_DIR = Path(os.environ.get("SECOM_DIR", ROOT.parent / "_downloads" / "secom"))
RESULTS = ROOT / "public" / "results.json"

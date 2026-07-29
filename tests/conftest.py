import sys
from pathlib import Path

# Tests import `src.*`, so the repository root must be importable regardless of where
# pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

# Add the src/ folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

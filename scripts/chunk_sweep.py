"""Thin shim — the implementation lives in rag_eval.cli.chunk_sweep (installable package)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_eval.cli.chunk_sweep import main  # noqa: E402

if __name__ == "__main__":
    main()

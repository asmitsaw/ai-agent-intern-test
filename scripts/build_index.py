"""
scripts/build_index.py — One-time script to chunk and embed the knowledge base.

Usage:
    python scripts/build_index.py

Run this once before starting the agent.  Re-running will wipe and rebuild.
"""
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.rag import build_index

if __name__ == "__main__":
    print("Building knowledge-base index...")
    count = build_index(verbose=True)
    print(f"\nDone. {count} chunks ready in the vector store.")

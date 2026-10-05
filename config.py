import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Relative paths resolve against the repo root, so scripts and notebooks share one DB.
CHROMA_DB_PATH = str(ROOT / os.getenv("CHROMA_DB_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kpmg_papers")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

#!/usr/bin/env python3
"""Audit chunk JSONL/JSON files before embedding or indexing.

Usage:
    python scripts/audit_chunks.py --chunks chunks/chunks.jsonl
    python scripts/audit_chunks.py --chunks chunks/chunks.jsonl --report reports/chunk_audit.json

Required fields follow the current ingest.py contract:
    chunk_id, paper_id, section_label, chunk_text

This script is read-only: it does not modify chunks or rebuild any vector index.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("chunk_id", "paper_id", "section_label", "chunk_text")
OPTIONAL_METADATA = (
    "title",
    "section_path",
    "page_start",
    "page_end",
    "chunk_index",
)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            chunks = []
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Line {line_no} is not a JSON object.")
                chunks.append(item)
            return chunks

        data = json.load(handle)
        if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
            raise ValueError("JSON input must be a list of chunk objects.")
        return data


def audit(chunks: list[dict[str, Any]], short_chars: int, long_chars: int) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    lengths: list[int] = []
    ids: Counter[str] = Counter()
    papers: Counter[str] = Counter()
    sections: Counter[str] = Counter()
    missing_optional: Counter[str] = Counter()

    def issue(index: int, chunk: dict[str, Any], kind: str, detail: str) -> None:
        problems.append({
            "chunk_index_in_file": index,
            "chunk_id": chunk.get("chunk_id"),
            "paper_id": chunk.get("paper_id"),
            "kind": kind,
            "detail": detail,
        })

    for index, chunk in enumerate(chunks):
        for field in REQUIRED_FIELDS:
            if field not in chunk:
                issue(index, chunk, "missing_required_field", field)
        for field in OPTIONAL_METADATA:
            if field not in chunk or chunk.get(field) in (None, ""):
                missing_optional[field] += 1

        chunk_id = chunk.get("chunk_id")
        paper_id = chunk.get("paper_id")
        section = chunk.get("section_path") or chunk.get("section_label")
        if chunk_id:
            ids[str(chunk_id)] += 1
        if paper_id:
            papers[str(paper_id)] += 1
        if section:
            sections[str(section)] += 1

        text = chunk.get("chunk_text")
        if not isinstance(text, str):
            issue(index, chunk, "invalid_text_type", "chunk_text must be a string")
            continue

        cleaned = text.strip()
        lengths.append(len(cleaned))
        if not cleaned:
            issue(index, chunk, "empty_text", "chunk_text is empty or whitespace only")
        elif len(cleaned) < short_chars:
            issue(index, chunk, "short_chunk_review", f"{len(cleaned)} characters (< {short_chars})")
        elif len(cleaned) > long_chars:
            issue(index, chunk, "long_chunk_review", f"{len(cleaned)} characters (> {long_chars})")

        if isinstance(chunk.get("page_start"), (int, float)) and isinstance(chunk.get("page_end"), (int, float)):
            if chunk["page_start"] < 1 or chunk["page_end"] < chunk["page_start"]:
                issue(index, chunk, "invalid_page_range",
                      f"page_start={chunk['page_start']}, page_end={chunk['page_end']}")

        if "section_path" in chunk and isinstance(chunk.get("section_path"), str):
            if not chunk["section_path"].strip():
                issue(index, chunk, "empty_section_path", "section_path is present but empty")

    for duplicate_id, count in ids.items():
        if count > 1:
            problems.append({
                "chunk_index_in_file": None,
                "chunk_id": duplicate_id,
                "paper_id": None,
                "kind": "duplicate_chunk_id",
                "detail": f"appears {count} times",
            })

    length_summary = None
    if lengths:
        ordered = sorted(lengths)
        length_summary = {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 1),
        }

    kind_counts = Counter(item["kind"] for item in problems)
    return {
        "chunk_count": len(chunks),
        "unique_chunk_ids": len(ids),
        "duplicate_chunk_id_count": sum(1 for count in ids.values() if count > 1),
        "paper_count": len(papers),
        "section_count": len(sections),
        "text_length_chars": length_summary,
        "missing_or_empty_optional_metadata_counts": dict(missing_optional),
        "issues_by_kind": dict(kind_counts),
        "issues": problems,
        "notes": [
            "Short/long chunk findings are review flags, not automatic failures.",
            "Page numbering conventions can vary by source; verify flagged page ranges against the PDF.",
            "This audit does not assess semantic relevance or embedding quality.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only audit of research-paper chunks.")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl or a JSON list.")
    parser.add_argument("--report", help="Optional path to write the full JSON report.")
    parser.add_argument("--short-chars", type=int, default=100,
                        help="Flag chunks shorter than this many characters (default: 100).")
    parser.add_argument("--long-chars", type=int, default=3000,
                        help="Flag chunks longer than this many characters (default: 3000).")
    args = parser.parse_args()

    path = Path(args.chunks)
    if not path.is_file():
        parser.error(f"Chunk file not found: {path}")
    if args.short_chars < 0 or args.long_chars <= args.short_chars:
        parser.error("--long-chars must be greater than --short-chars, and --short-chars cannot be negative.")

    try:
        chunks = load_chunks(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error loading chunks: {exc}\n")

    report = audit(chunks, args.short_chars, args.long_chars)
    print(json.dumps({k: v for k, v in report.items() if k != "issues"}, indent=2))
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nFull report written to: {out}")

    hard_errors = sum(
        count for kind, count in report["issues_by_kind"].items()
        if kind in {"missing_required_field", "invalid_text_type", "empty_text", "duplicate_chunk_id", "invalid_page_range"}
    )
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

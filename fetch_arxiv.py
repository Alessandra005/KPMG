import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone

import arxiv

import config

DATA_DIR = config.ROOT / "data"
METADATA_PATH = DATA_DIR / "arxiv_metadata.jsonl"


def safe_filename(paper: arxiv.Result) -> str:
    """Match the existing '<id>v<n>_<title>.pdf' convention the chunking notebook reads."""
    title = re.sub(r"[^\w\s'-]", "", paper.title)
    title = re.sub(r"\s+", " ", title).strip()[:80]
    return f"{paper.get_short_id()}_{title}.pdf"


def known_paper_ids() -> set[str]:
    """Versionless IDs already on disk, so re-runs and new versions aren't re-downloaded."""
    return {p.name.split("_", 1)[0].rsplit("v", 1)[0] for p in DATA_DIR.glob("*.pdf")}


def build_query(category: str, days: int, keywords: list[str]) -> str:
    query = f"cat:{category}"
    if keywords:
        query += " AND (" + " OR ".join(f'abs:"{k}"' for k in keywords) + ")"
    if days:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        query += f" AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    return query


def fetch(query: str, max_results: int, download: bool) -> list[dict]:
    # arXiv asks for ~1 request per 3 seconds; the client enforces this via delay_seconds.
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    DATA_DIR.mkdir(exist_ok=True)
    seen = known_paper_ids()
    new = []
    for paper in client.results(search):
        paper_id = paper.get_short_id().rsplit("v", 1)[0]
        if paper_id in seen:
            continue
        if download:
            paper.download_pdf(dirpath=str(DATA_DIR), filename=safe_filename(paper))
        seen.add(paper_id)
        new.append(
            {
                "paper_id": paper_id,
                "version": paper.get_short_id().rsplit("v", 1)[1],
                "title": " ".join(paper.title.split()),
                "abstract": " ".join(paper.summary.split()),
                "authors": [a.name for a in paper.authors],
                "categories": paper.categories,
                "published": paper.published.isoformat(),
                "pdf_url": paper.pdf_url,
            }
        )
        print(f"{'downloaded' if download else 'found'} {paper.get_short_id()}: {paper.title[:70]}")
    return new


def main():
    parser = argparse.ArgumentParser(description="Pull recent arXiv papers into data/.")
    parser.add_argument("--category", default="cs.AI")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--days", type=int, default=2, help="Only papers submitted in the last N days (0 = no limit)")
    parser.add_argument("--keyword", action="append", default=[], help="Abstract keyword filter; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="List matches without downloading PDFs")
    args = parser.parse_args()

    query = build_query(args.category, args.days, args.keyword)
    print(f"Query: {query}")
    try:
        new = fetch(query, args.max_results, download=not args.dry_run)
    except arxiv.ArxivError as e:
        sys.exit(f"Error: arXiv request failed: {e}")

    if new and not args.dry_run:
        with open(METADATA_PATH, "a", encoding="utf-8") as f:
            for record in new:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(new)} new paper(s).")


if __name__ == "__main__":
    main()

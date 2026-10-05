import chromadb

import config

BATCH_SIZE = 500


def _client():
    return chromadb.PersistentClient(path=config.CHROMA_DB_PATH)


def get_collection():
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": config.EMBEDDING_MODEL},
    )


def check_embedding_model(collection) -> None:
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != config.EMBEDDING_MODEL:
        raise RuntimeError(
            f"Collection '{collection.name}' was built with {built_with!r} but EMBEDDING_MODEL is "
            f"{config.EMBEDDING_MODEL!r}. Re-run ingestion to rebuild it."
        )


def chunk_metadata(chunk: dict, titles: dict) -> dict:
    # Chroma rejects None values, so every field falls back to a concrete default.
    return {
        "paper_id": chunk["paper_id"],
        "title": titles.get(chunk["paper_id"], ""),
        "section_label": chunk["section_label"],
        "section_path": chunk.get("section_path", chunk["section_label"]),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "page_start": int(chunk.get("page_start", 0)),
        "page_end": int(chunk.get("page_end", 0)),
    }


def ingest_chunks(chunks: list[dict], embeddings: list[list[float]], titles: dict | None = None):
    titles = titles or {}
    collection = get_collection()
    if (collection.metadata or {}).get("embedding_model") != config.EMBEDDING_MODEL:
        # Vectors from different models (and dimensions) can't share one index, so rebuild.
        _client().delete_collection(config.COLLECTION_NAME)
        collection = get_collection()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=[c["chunk_text"] for c in batch],
            metadatas=[chunk_metadata(c, titles) for c in batch],
        )
    return collection

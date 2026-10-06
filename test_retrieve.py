from retrieve import retrieve

# TEST 1
results = retrieve(
    "How does active data collection affect travel behavior predictions?",
    k=5
)

print("\nTEST 1")
for r in results:
    print(
        r["chunk_id"],
        "|", r["paper_id"],
        "|", r["section_label"],
        "|", r["page_start"], "-", r["page_end"],
        "| score:", r["score"]
    )


# TEST 2
results = retrieve(
    "How does active data collection affect travel behavior predictions?",
    k=1
)

print("\nTEST 2")
print(results[0])


# TEST 3
results = retrieve(
    "What are common limitations of small-sample AI studies?",
    k=5
)

print("\nTEST 3")
for r in results:
    print(
        r["chunk_id"],
        "|", r["paper_id"],
        "|", r["section_label"],
        "|", r["page_start"], "-", r["page_end"],
        "| score:", r["score"]
    )
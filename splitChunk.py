import json
import sqlite3

CHUNKS_FILE = "apmsmeone.ap.gov.in_2026-01-22_08-44-04/Citta_Chunks_apmsmeone.json"
DB_FILE = "apmsmeone.ap.gov.in_2026-01-22_08-44-04/chunks.db"

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    chunk_index INTEGER,
    title TEXT,
    summary TEXT,
    propositions TEXT,
    canonical_text TEXT
)
""")

with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)

inserted = 0

for _, chunk in chunks.items():
    cur.execute("""
        INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chunk["chunk_id"],
        chunk.get("chunk_index"),
        chunk.get("title"),
        chunk.get("summary"),
        json.dumps(chunk.get("propositions", [])),
        chunk.get("canonical_text")
    ))
    inserted += 1

conn.commit()
conn.close()

print(f"✅ {inserted} chunks inserted into chunks.db")

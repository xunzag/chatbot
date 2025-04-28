import cohere
import faiss
import json
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DATA_PATH = "data/parsed_chunks.json"
INDEX_PATH = "data/faiss_index.idx"
CHUNK_META_PATH = "data/chunk_meta.json"

EMBED_MODEL = "embed-english-v3.0"  # Or use multilingual-v3.0
BATCH_SIZE = 50  # Process 50 chunks at a time
RATE_LIMIT_DELAY = 1  # 1 second delay between batches

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_chunks(chunks):
    co = cohere.Client(COHERE_API_KEY)
    all_embeddings = []
    
    # Process chunks in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        try:
            batch_embeddings = co.embed(
                texts=texts,
                model=EMBED_MODEL,
                input_type="search_document"
            ).embeddings
            all_embeddings.extend(batch_embeddings)
            
            # Add delay between batches to respect rate limit
            if i + BATCH_SIZE < len(chunks):
                time.sleep(RATE_LIMIT_DELAY)
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            # If we hit rate limit, wait longer and retry
            if "rate limit" in str(e).lower():
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                i -= BATCH_SIZE  # Retry this batch
                continue
            raise e
    
    return np.array(all_embeddings).astype("float32")

def save_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index to {index_path}")
    return index

def save_chunk_meta(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    print("📦 Loading chunks...")
    chunks = load_chunks(DATA_PATH)

    print("🔍 Embedding chunks with Cohere...")
    embeddings = embed_chunks(chunks)

    print("💾 Saving index and metadata...")
    save_faiss_index(embeddings, INDEX_PATH)
    save_chunk_meta(chunks, CHUNK_META_PATH)

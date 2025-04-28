import fitz  # PyMuPDF
import os
import json

# === CONFIG ===
PDF_DIR = "pdfs"
OUTPUT_FILE = "data/parsed_chunks.json"
CHUNK_SIZE = 800  # character-based chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < chunk_size:
            current += line + " "
        else:
            chunks.append(current.strip())
            current = line + " "
    if current:
        chunks.append(current.strip())
    return chunks

def process_all_pdfs():
    os.makedirs("data", exist_ok=True)
    all_chunks = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            print(f"Processing {filename}...")
            raw_text = extract_text_from_pdf(path)
            chunks = chunk_text(raw_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": filename,
                    "chunk_id": f"{filename}_{i}",
                    "text": chunk
                })
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\n✅ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_pdfs()

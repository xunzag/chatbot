import cohere
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from model.generate import generate_response

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_PATH = "data/faiss_index.idx"
CHUNK_META_PATH = "data/chunk_meta.json"
EMBED_MODEL = "embed-english-v3.0"  # Or "multilingual-v3.0"

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    tokens = nltk.word_tokenize(query.lower())
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(cleaned_tokens)

def embed_query(query):
    preprocessed_query = preprocess_query(query)
    co = cohere.Client(COHERE_API_KEY)
    embedding = co.embed(texts=[preprocessed_query], model=EMBED_MODEL, input_type="search_query").embeddings[0]
    return np.array(embedding).astype("float32")

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_chunk_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        chunk_meta = json.load(f)
    
    return chunk_meta

def retrieve(query, index, chunk_meta, k=5):
    query_embedding = embed_query(query)
    D, I = index.search(np.array([query_embedding]), k)

    # Extracting relevant chunks
    relevant_chunks = []
    for idx in I[0]:
        chunk_info = chunk_meta[idx]
        chunk_text = chunk_info['text']
        relevant_chunks.append(chunk_text)

    return relevant_chunks

def generate_response(relevant_chunks, query):
    # Combine the relevant chunks to form a detailed answer
    context = " ".join(relevant_chunks)
    
    co = cohere.Client(COHERE_API_KEY)
    response = co.generate(
        model="command",  # Ensure you are using the correct model
        prompt=f"Given the following context, answer the question: {query}\n\nContext: {context}\nAnswer:",
        max_tokens=150,
        temperature=0.4,
    ).generations[0].text.strip()

    return response

if __name__ == "__main__":
    print("📦 Loading FAISS index and chunk metadata...")
    index = load_faiss_index(INDEX_PATH)
    chunk_meta = load_chunk_meta(CHUNK_META_PATH)

    query = "What are the safety requirements for traffic controllers at night in Queensland?"  # Example query
    print(f"🔍 Searching for: {query}")

    print("🔑 Retrieving relevant information...")
    relevant_chunks = retrieve(query, index, chunk_meta)
    
    print(f"💡 Retrieved {len(relevant_chunks)} relevant chunks.")
    
    print("\n🔍 Generating response...")
    response = generate_response(relevant_chunks, query)
    print(f"💡 Response: \n{response}")

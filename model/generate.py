import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def generate_response(question, retrieved_chunks):
    # Combine chunks into one context block
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks if chunk['text'].strip()])

    # Create a clean prompt
    prompt = f"""
You are a helpful assistant specialized in traffic management regulations. Use the provided context to answer the question below in a clear, professional, and well-structured manner. If the context does not contain an answer, just say you don't know.

Context:
{context}

Question: {question}
Answer:
"""

    # Generate response using Cohere
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=300,
        temperature=0.4,
    )

    return response.generations[0].text.strip()

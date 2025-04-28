import cohere
import numpy as np
import faiss
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for local development

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Cohere
api_key = 'KXOoiAr2pDbhgnNNzg7NwJ8ZTPGhH5VIf00Av9q1'
cohere_client = cohere.Client(api_key)

# Load your chunks data (replace with your actual path)
chunk_meta_path = os.path.join(BASE_DIR, 'data', 'chunk_meta.json')
with open(chunk_meta_path, 'r', encoding='utf-8') as f:
    your_data = json.load(f)

# Load FAISS index
index_file_path = os.path.join(BASE_DIR, 'data', 'faiss_index.idx')
index = faiss.read_index(index_file_path)

# Embedding function using Cohere
def embedding_function(text):
    response = cohere_client.embed(
        texts=[text],
        model="embed-english-v3.0",  # or whatever model you're using
        input_type="search_query"    # this is required now
    )
    return response.embeddings[0]


# FAISS search
def search_faiss(query, k=5):
    query_embedding = np.array([embedding_function(query)], dtype='float32')
    distances, indices = index.search(query_embedding, k)
    return indices[0]  # Return the top-level list of indices

# Generate response using Cohere
def generate_response(context, query):
    # Detect query type and adjust personality
    is_technical = any(word in query.lower() for word in ['regulation', 'requirement', 'standard', 'procedure', 'guideline'])
    is_safety = any(word in query.lower() for word in ['safety', 'hazard', 'risk', 'danger', 'emergency'])
    
    # Create a dynamic personality based on query type
    personality = """You are TrafficGPT, a knowledgeable and charismatic traffic management expert with a great sense of humor.
    While maintaining professionalism, you can:
    1. Use witty traffic-related puns when appropriate
    2. Add engaging examples
    3. Be conversational and friendly
    4. Use emojis sparingly but effectively
    5. Break down complex information into digestible parts
    
    However, when discussing safety or technical regulations, prioritize clarity and accuracy over humor."""

    # Craft the prompt based on query type
    if is_technical or is_safety:
        prompt = f"""Based on the following context, provide a detailed, accurate, and well-structured response.
        Be thorough and ensure no important information is omitted.

        Context: {context}

        Question: {query}

        Please structure your response with:
        1. A clear introduction
        2. Main points with proper explanation
        3. Examples or clarifications where needed
        4. A complete conclusion
        
        Remember to maintain a professional tone while being engaging."""
    else:
        prompt = f"""Based on the following context, provide a friendly and informative response.
        Feel free to use appropriate humor and make it engaging while staying accurate.

        Context: {context}

        Question: {query}

        Start with a friendly opener, then:
        1. Address the main points
        2. Add relevant examples or analogies
        3. Include a light-hearted comment or pun if appropriate
        4. End with a helpful conclusion

        Keep it informative but conversational!"""

    try:
        response = cohere_client.generate(
            model='command',
            prompt=f"{personality}\n\n{prompt}",
            max_tokens=500,  # Increased for more complete responses
            temperature=0.7,
            p=0.9,  # Added nucleus sampling
            frequency_penalty=0.3,  # Reduce repetition
            presence_penalty=0.3,  # Encourage diverse content
            stop_sequences=["\n\n\n"]  # Prevent excessive newlines
        )
        
        generated_text = response.generations[0].text.strip()
        
        # Post-process the response
        if len(generated_text.split()) < 20:  # If response is too short
            # Generate a follow-up
            followup = cohere_client.generate(
                model='command',
                prompt=f"The previous response was: {generated_text}\n\nPlease expand on this with more details and examples.",
                max_tokens=200,
                temperature=0.6
            ).generations[0].text.strip()
            generated_text = f"{generated_text}\n\n{followup}"
        
        # Ensure response ends properly
        if not any(generated_text.strip().endswith(char) for char in '.!?'):
            generated_text += '.'
            
        return generated_text

    except Exception as e:
        print(f"Error in response generation: {e}")
        # Fallback to a simpler response
        return f"Based on the available information: {context[:200]}..."

# API endpoint
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Step 1: Search in FAISS
        indices = search_faiss(query)

        # Step 2: Get matching chunks
        relevant_chunks = [your_data[i] for i in indices]

        # Step 3: Generate answer
        context = " ".join(chunk["text"] for chunk in relevant_chunks)
        response = generate_response(context, query)

        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

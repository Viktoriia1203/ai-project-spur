from openai import OpenAI
import numpy as np

# 1. Initialize client
client = OpenAI(api_key="YOUR_API_KEY")

# 2. Sample documents (your knowledge base)
documents = [
    "I love artificial intelligence",
    "Python is a powerful programming language",
    "Machine learning is the future",
    "Cats are cute animals"
]

# 3. Function to get embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 4. Create embeddings for all documents
doc_embeddings = [get_embedding(doc) for doc in documents]

# 5. Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 6. Search function
def search(query):
    query_embedding = get_embedding(query)

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    best_match_index = np.argmax(similarities)
    return documents[best_match_index]

# 7. Test query
query = "Tell me about artificial intelligence"
result = search(query)

print("Query:", query)
print("Best match:", result)

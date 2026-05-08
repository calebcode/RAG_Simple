import chromadb
import ollama

# initialize chroma client
client = chromadb.PersistentClient(path="my_knowledge_base")
collection = client.get_or_create_collection(name="tech_docs")

def add_document(text, doc_id, metadata):
    # generate embedding using your local ollama model
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    embedding = response["embedding"]

    # store in the vector db
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )
    print(f"Added: {doc_id}")

def query_knowlege_base(question):
    # vectorize the question
    query_emb = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]

    # query chroma for the top two most relevant chunks
    results = collection.query(query_embeddings=[query_emb], n_results=2)

    context = "\n".join(results['documents'][0])
    return context

# example usage
if __name__ == "__main__":
    # simulate adding a piece of info
    add_document(
        "The project 'phoenix' uses a 10Gbps backbone and Pi-hole for DNS filtering.",
        "doc_001",
        {"source": "internal_memo"}
    )

    # query it
    relevant_context = query_knowlege_base("What is the networking setup for project phoenix?")
    print(f"\nRetrieved Context:\n{relevant_context}")
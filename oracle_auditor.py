import ollama
import chromadb

def get_oracle_advice(query):
    client = chromadb.PersistentClient(path="my_knowledge_base")
    collection = client.get_collection(name="tech_docs")

    query_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_emb], n_results=3, include=["documents","metadatas","distances"])

    context = ""
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i] # this is for observability

        print(f"--- [LOG] Debugging distances for chunk {i}: {dist:.4f} ---")

        context += f"\n[Source: {meta['source']} p.{meta['page']} | Distance: {dist:.4f}]\n{text}\n"

    return context

def audit_with_oracle(file_path):
    with open(file_path, "r") as f:
        code = f.read()

    # get relevant coding standards from knowledge base
    context = get_oracle_advice("coding standards and security best practices")

    prompt = f"""
    Use the following Reference Context to audit the Code.

    Reference Context:
    {context}

    Code to Audit:
    {code}
    """

    response = ollama.chat(model='qwen2.5-coder:7b', messages=[
        {'role': 'system', 'content': 'You are a senior auditor. Ground your advice in the provided Reference Context.'},
        {'role': 'user', 'content': prompt}
    ])
    print(response['message']['content'])

if __name__ == "__main__":
    audit_with_oracle("ingest.py")
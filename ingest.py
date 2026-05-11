import os
from pypdf import PdfReader
import chromadb
import ollama

# recursive chunking logic
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    
    return chunks

def ingest_docs(folder_path):
    client = chromadb.PersistentClient(path="my_knowledge_base")
    collection = client.get_or_create_collection(name="tech_docs", metadata={"hnsw:space": "cosine"})

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                page_chunks = chunk_text(text)

                for i, chunk in enumerate(page_chunks):
                    chunk_id = f"{filename}_p{page_num}_c{i}"
                    embedding = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]

                    collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas={"source": filename, "page": page_num + 1}
                    )

    print("Ingestion complete.")

if __name__ == "__main__":
    # make sure you create docs folder and put files in it (or do as you wish, i don't care)
    ingest_docs("docs")
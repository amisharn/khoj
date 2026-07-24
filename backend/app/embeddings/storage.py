from backend.app.embeddings.encoder import encode
import pickle
from pathlib import Path

DB_PATH = Path("backend/data/processed/embeddings.pkl")

def generate_document_embeddings(processed_text):
    document_embeddings = []
    for document_name,text in processed_text.items():
        embedding = encode(text)
        document_embeddings.append({"document_name": document_name, "embedding": embedding, "text": text })
    return document_embeddings


def save_embeddings(document_embeddings):
    DB_PATH.parent.mkdir(parents = True, exist_ok = True)
    with open(DB_PATH,"wb") as file:
        pickle.dump(document_embeddings,file)

def load_embeddings():
    if not DB_PATH.exists():
        return []
    with open (DB_PATH,"rb") as file:
        return pickle.load(file)

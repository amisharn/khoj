from backend.app.embeddings.storage import generate_document_embeddings,save_embeddings,load_embeddings
from backend.app.loaders.document_loader import load_documents, extract_text
from backend.app.processing.preprocess import preprocess_docs
from pathlib import Path

docs = load_documents(Path("backend/data/raw"))

document_text = extract_text(docs)


embeddings = generate_document_embeddings(document_text)
save_embeddings(embeddings)
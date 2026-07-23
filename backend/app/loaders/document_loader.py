from pathlib import Path
from pypdf import PdfReader

folder = Path("backend/data/raw")

def load_documents(folder: Path):
    docs = {}
    for item in folder.iterdir():
        if item.suffix.lower() == ".pdf":
            reader = PdfReader(item)
            document_text = ""
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            document_text = "\n".join(pages)
            docs[item.stem] = {"text" : document_text, "path" : str(item), "pages": len(reader.pages)}
    return docs

def extract_text(docs):
    document_text ={}
    for doc_name, values in docs.items():
        document_text[doc_name] = values["text"]
    return document_text

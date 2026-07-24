from backend.app.embeddings.encoder import encode
from backend.app.embeddings.storage import load_embeddings

from sklearn.metrics.pairwise import cosine_similarity


def similarity_scores(query_encoding, document_encoding):
    cosine_similarity_value = []
    query = query_encoding.reshape(1,-1)
    for document in document_encoding:
        document_embedding = document["embedding"].reshape(1,-1)
        similarity = cosine_similarity(query,document_embedding).item()
        cosine_similarity_value.append({"document_name":document["document_name"],"score": similarity ,"text": document["text"]})
    return cosine_similarity_value

def semantic_search_result(cosine_similarity_values, k = 1):
    ranked = sorted(cosine_similarity_values,key = lambda x: x["score"] ,reverse=True)
    return ranked [:k]


def semantic_search(query):
    query_encode = encode(query)

    document_encoding = load_embeddings()

    similarity_score = similarity_scores(query_encode,document_encoding)

    results = semantic_search_result(similarity_score)

    return results






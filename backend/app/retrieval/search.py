from backend.app.retrieval.cosine import cosine_similarity


def search(query,index):
        return index.get(query, [])

def similarity_scores(query_vector, document_vector):
    cosine_similarity_value = {}
    for document,document_vector_value in document_vector.items():
        cosine_similarity_value[document] = (cosine_similarity(query_vector,document_vector_value))
    return cosine_similarity_value

def result(vector, k = 1):
    ranked = sorted(vector.items(),key = lambda x: x[1] ,reverse=True)
    return ranked [:k]

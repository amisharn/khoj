from loaders.document_loader import load_documents, extract_text

from indexing.inverted_index import inverted_index

from processing.preprocess import preprocess_docs,preprocess_query

from weighting.tfidf import term_frequency,document_frequency,idf,tf_idf

from weighting.vectorizer import build_vocabulary,vectorize

from retrieval.search import search

from retrieval.cosine import dot_product,magnitude,cosine_similarity

from pathlib import Path

docs = load_documents(Path("backend/data/raw"))

document_text = extract_text(docs)

processed_text = preprocess_docs(document_text)

index = inverted_index(processed_text)

count = (term_frequency(processed_text))

df = document_frequency(count)

idf_val = idf(document_text,df)

tfidf = tf_idf(count,idf_val)

vocab = build_vocabulary(tfidf)

document_vector = vectorize(tfidf,vocab)

query = input("Enter your query: ")

processed_query = preprocess_query(query)

query_count = term_frequency(processed_query)

query_tfidf = tf_idf(query_count,idf_val)

query_vector = vectorize(query_tfidf,vocab)["query"]

def similarity_scores(query_vector, document_vector):
    cosine_similarity_value = {}
    for document,document_vector_value in document_vector.items():
        cosine_similarity_value[document] = (cosine_similarity(query_vector,document_vector_value))
    return cosine_similarity_value

def result(vector, k = 1):
    ranked = sorted(vector.items(),key = lambda x: x[1] ,reverse=True)
    return ranked [:k]

scores = similarity_scores(query_vector,document_vector)
results = result(scores)

print("Top Results:")

for rank, (doc,score) in enumerate(results,start=1):
    print (f"{rank}. {doc} (score:{score:.4f})")
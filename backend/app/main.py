from backend.app.loaders.document_loader import load_documents, extract_text

from backend.app.indexing.inverted_index import inverted_index

from backend.app.processing.preprocess import preprocess_docs,preprocess_query

from backend.app.weighting.tfidf import term_frequency,document_frequency,idf,tf_idf

from backend.app.weighting.vectorizer import build_vocabulary,vectorize

from backend.app.retrieval.search import search, similarity_scores, result

from backend.app.retrieval.semantic_search import semantic_search

from pathlib import Path

from backend.app.embeddings.encoder import encode

from backend.app.embeddings.storage import generate_document_embeddings,save_embeddings,load_embeddings



docs = load_documents(Path("backend/data/raw"))

document_text = extract_text(docs)

processed_text = preprocess_docs(document_text)

index = inverted_index(processed_text)

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

print(f"\n------------------------------------------------\n")

print("\n Results Key-Word Search: ")
results = search(query, index)

for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc}")

print(f"\n------------------------------------------------\n")


scores = similarity_scores(query_vector,document_vector)
results = result(scores)

print("\n Top Results TF-IDF:")

for rank, (doc,score) in enumerate(results,start=1):
    print (f"\n {rank}. {doc} (score:{score:.4f})")

print(f"\n------------------------------------------------\n")


result_semantic_search = semantic_search(query)

print(f"\n Top k Results Semantic Search:")

for rank,doc in enumerate(result_semantic_search, start=1):
    print(f"\n {rank}. {doc['document_name']} (score:{doc['score']:.04f})")

print(f"\n------------------------------------------------\n")


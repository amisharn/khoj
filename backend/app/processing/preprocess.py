import re

def preprocess_docs(docs):
    preprocessed_text = {}
    for doc,text in docs.items():
        preprocessed_text[doc] = []
        clean_text = re.sub(r'[^\w\s]','',text)
        words = clean_text.split()
        preprocessed_text[doc] = [word.lower() for word in words]
    return preprocessed_text

def preprocess_query(query):
    query_return = {}
    query_terms = query.split()
    query_return["query"] = []
    for word in query_terms:
        word = re.sub(r'[^\w\s]','',word)
        word = word.lower()
        query_return["query"].append(word)
    return query_return
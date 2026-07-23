def build_vocabulary(tfidf):
    vocab = set()
    for doc,values in tfidf.items():
        for word,value in values.items():
            vocab.add(word)
    return sorted(vocab)

def vectorize(tfidf,vocabulary):
    vector = {}
    for doc,values in tfidf.items():
        vector[doc] = []
        for word in vocabulary:
            if word in values:
                vector[doc].append(values[word])
            else:
                vector[doc].append(0)
    return vector

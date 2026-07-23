def inverted_index(preprocessed_text):
    index = {}
    for doc,text in preprocessed_text.items():
        for word in text:
            if word not in index:
                index[word] = []
            if doc not in index[word]:
                index[word].append(doc)
    return index

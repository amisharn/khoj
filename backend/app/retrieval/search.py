def search(query,index):
    results = []
    for word in query:
        results.append(index.get(word,[]))
    return results
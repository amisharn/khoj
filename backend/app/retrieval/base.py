import re
import math

docs = {
"doc1" : "Python Python is fun.",
"doc2" : "I love love Python.",
"doc3" : "FastAPI builds APIs"
}

def preprocess_docs(docs):
    preprocessed_text = {}
    for doc,text in docs.items():
        preprocessed_text[doc] = []
        clean_text = re.sub(r'[^\w\s]','',text)
        words = clean_text.split()
        preprocessed_text[doc] = [word.lower() for word in words]
    return preprocessed_text

   
def inverted_index(preprocessed_text):
    index = {}
    for doc,text in preprocessed_text.items():
        for word in text:
            if word not in index:
                index[word] = []
            if doc not in index[word]:
                index[word].append(doc)
    return index


def preprocess_query(query):
    query_return = []
    query_terms = query.split()
    for query in query_terms:
        query = re.sub(r'[^\w\s]','',query)
        query = query.lower()
        query_return.append(query)
    return query_return
    


def search(query,index):
    return index.get(query,[])

def term_frequency(preprocessed_docs):
    count = {}
    for doc,text in preprocessed_docs.items():
        total_words = 0
        count[doc] = {}
        for word in text:
            if word not in count[doc]:
                count[doc][word] = 1
            else:
                count[doc][word]+=1
        total_words = len (text)
        for word in count[doc]:
            count[doc][word] = count[doc][word]/total_words
    return count

def document_frequency(count):
    df = {}
    for doc,term_frequency in count.items():
        for word in term_frequency:
            if word not in df:
                df[word] = 1
            else:
                df[word]+= 1
    return df

def idf(docs,df):
    idf_count = {}
    N = len(docs)
    for word,freq in df.items():
        idf_count[word] = math.log(N/freq)
    return idf_count

def tf_idf(tf,idf):
    tf_idf = {}
    for doc,words in tf.items():
        tf_idf[doc] = {}
        for word,occurence in words.items():
            tf_idf[doc][word] =  occurence * idf [word]
    return tf_idf


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


count = (term_frequency(preprocess_docs(docs)))

df = document_frequency(count)

idf_val = idf(docs,df)

tfidf = tf_idf(count,idf_val)

vocab = build_vocabulary(tfidf)

print(vocab)

print (vectorize(tfidf,vocab))


    
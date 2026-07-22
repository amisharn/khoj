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
    query = re.sub(r'[^\w\s]','',query)
    query = query.lower()
    return query
    


def search(query,index):
    return index.get(query,[])

def term_frequency(preprocessed_docs):
    count = {}
    for doc,text in preprocessed_docs.items():
        count[doc] = {}
        for word in text:
            if word not in count[doc]:
                count[doc][word] = 1
            else:
                count[doc][word]+=1
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

count = (term_frequency(preprocess_docs(docs)))

df = document_frequency(count)

idf_val = idf(docs,df)

print(idf_val)



    
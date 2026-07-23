import math

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
            if word in idf:
                tf_idf[doc][word] =  occurence * idf [word]
            else:
                tf_idf[doc][word] = 0
    return tf_idf
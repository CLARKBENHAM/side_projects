import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

"""Question: given that words follow zipfs law, 
When an inverted index is created what is the optimal encoding metric for the distance between documents?
assumption:
    word usage in document is uncorrelated
    word frequency is zipfs law
    documents ~equal length
    documents equally likely(any order for documents in linked list is equally efficent) [follows from previous 3]
"""
doc_size = 1000
num_docs = 5000
num_words = 10000#0
zipf_p = 1.5
log2 = lambda arr: np.floor(np.log(arr)/np.log(2))
log2C = lambda arr: np.floor(np.log(arr)/np.log(2)) + 1
                     
def gamma(x, cnts):
    "return bits to gamma encode array x"
    return 2*np.sum(log2C(x)*cnts)

def delta(x, cnts):
    "return bits to delta encode array x"
    return np.sum(cnts*(log2(x) + 2*log2(log2(x)+1) + 1))

def zeta(x, cnts):
    """return bits to zeta encode array x; continuing in naming scheme
    Let N = ⌊log2 X⌋; be the highest power of 2 in X, so 2^N ≤ X < 2^N+1.
    Let L = ⌊log2 N+1⌋ be the highest power of 2 in N+1, so 2^L ≤ N+1 < 2^L+1.
    Let M = ⌊log2 L+1⌋ be the highest power of 2 in L+1, so 2^M ≤ L+1 < 2^M+1.
    Write M zeros, followed by
    the M+1-bit binary representation of K+1, followed by
    the L+1-bit binary representation of N+1, followed by
    all but the leading bit (i.e. the last N bits) of X."""
    N = log2(x)
    L = log2(N+1)
    M = log2(L+1)
    return np.sum(cnts*(2*M + 1 + L + 1 + N)) #delta(log2(x), cnts) + np.sum((1 + log2(x))*cnts)

#print("Random Doc size: ", np.mean(ss.lognorm.rvs(loc=1, s = np.log(10), size = 1000)*200))
for doc_size in [50, 250, 1000, 5000]:
    index = np.zeros([num_words, num_docs])
    for doc in range(num_docs):
    #    doc_size = int(ss.lognorm.rvs(loc=1, s = np.log(10))*200)
        for word in (i for i in ss.zipf.rvs(zipf_p, size=doc_size) if i < num_words and i > 100):#takes out head words
            index[word, doc] = 1
    #%     
    word_ix, doc_ix = np.nonzero(index)
    doc_arrs = np.split(doc_ix, np.cumsum(np.unique(word_ix, return_counts=True)[1]))
    dist_indx = [np.diff(doc_arr) for doc_arr in doc_arrs]
    #%
    words, word_counts = np.unique(np.concatenate(dist_indx), return_counts=True)
    #plt.scatter(words, word_counts) 
    #% 
    print(f"Gamma encoding average bits per {doc_size} word doc: {gamma(words, word_counts)/num_docs}")
    print(f"Delta encoding average bits per {doc_size} word doc: {delta(words, word_counts)/num_docs}")
    print(f"Zeta encoding average bits per {doc_size} word doc: {zeta(words, word_counts)/num_docs}\n")

#%%  
"""Gamma encoding average bits per 50 word doc: 41.0072
Delta encoding average bits per 50 word doc: 33.8856
Zeta encoding average bits per 50 word doc: 38.1184

Gamma encoding average bits per 250 word doc: 184.0836
Delta encoding average bits per 250 word doc: 156.0386
Zeta encoding average bits per 250 word doc: 177.802

Gamma encoding average bits per 1000 word doc: 582.2884
Delta encoding average bits per 1000 word doc: 501.789
Zeta encoding average bits per 1000 word doc: 594.394

Gamma encoding average bits per 5000 word doc: 1789.6608
Delta encoding average bits per 5000 word doc: 1529.8896
Zeta encoding average bits per 5000 word doc: 1921.5268"""
import random
import pickle
import math
import sys
import numpy as np
from numpy import linalg as LA
import scipy.spatial.distance

AllVectors = pickle.load(open(sys.argv[1])) 

PredVectors = []

WordnetWords = []
for line in open(sys.argv[2]):
    WordnetWords.append(line[:-1])

for line in sys.stdin:
    thisLine = line.split()
    word = thisLine[0]
    vec = np.array(map(lambda x: float(x), thisLine[1:]))
    vec/=LA.norm(vec)
    PredVectors.append((word, vec))
    
VectorList = []

ResultsList = []

for hyper, predVector in PredVectors:
    VectorList = []
    actual = AllVectors[hyper]
    key = (math.acos(np.inner(actual, predVector)),"_PRED")
    VectorList.append(key)

    CompareWords = random.sample(WordnetWords,99)
    #CompareWords = []
#    while len(CompareWords)<2:
#        word = random.choice(WordnetWords)
#        if word in AllVectors.keys() and not word==hyper:
#            CompareWords.append(word)

    for word in CompareWords:
        if word in AllVectors.keys() and not word==hyper:
            Vector = AllVectors[word]
            VectorList.append( (\
                math.acos(np.inner(actual,Vector)),word\
                ))

    VectorList = sorted(VectorList)
#    print VectorList
#    print hyper, VectorList.index(key)
    ResultsList.append(VectorList.index(key))

#print ResultsList
print float(sum(ResultsList))/float(len(ResultsList))

import pickle
import math
import sys
import numpy as np
from numpy import linalg as LA
import scipy.spatial.distance

AllVectors = pickle.load(open(sys.argv[1])) 

PredVectors = {}

for line in sys.stdin:
    thisLine = line.split()
    word = thisLine[0]
    PredVectors[word] = np.array(map(lambda x: float(x), thisLine[1:]))
    PredVectors[word]/=LA.norm(PredVectors[word])

VectorList = []

for hyper, predVector in PredVectors.iteritems():
    VectorList = []
    actual = AllVectors[hyper]
    key = (math.acos(np.inner(actual, predVector)),"_PRED")
    VectorList.append(key)
    for word, Vector in AllVectors.iteritems():
        if word==hyper:
            continue
        VectorList.append( (\
            math.acos(np.inner(actual,Vector)),word\
            ))
    VectorList = sorted(VectorList)
    print hyper, VectorList.index(key)


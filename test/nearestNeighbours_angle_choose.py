import random
import pickle
import math
import sys
import numpy as np
from numpy import linalg as LA
import scipy.spatial.distance

AllVectors = pickle.load(open(sys.argv[1]))

PredVectors = []

for line in sys.stdin:
    thisLine = line.split()
    word = thisLine[0]
    vec = np.array(map(lambda x: float(x), thisLine[1:]))
    vec/=LA.norm(vec)
    PredVectors.append((word, vec))
    
ResultsList = []

for hyper, predVector in PredVectors:
    actual = AllVectors[hyper]
    key = (math.acos(np.inner(actual, predVector)),"_PRED")

    compareWord = random.choice(AllVectors.keys())
    otherVector = AllVectors[compareWord]
    otherKey = (math.acos(np.inner(actual,otherVector)),"_OTHER")

    VectorList = [ (math.acos(np.inner(actual, vector)),word) for word, vector in AllVectors.iteritems() if not word==hyper]

    VectorList.append(key)
    VectorList.append(otherKey)
    VectorList = sorted(VectorList)
    ResultsList.append((VectorList.index(key),VectorList.index(otherKey)))

for (a,b) in ResultsList:
    sys.stdout.write(str(a)+"\t"+str(b)+"\n")

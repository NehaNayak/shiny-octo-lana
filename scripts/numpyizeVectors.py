import sys
import numpy as np
import pickle

Vectors = {}

for line in sys.stdin:
	thisLine = line.split()
	thisVector = np.array(map(lambda x: float(x), thisLine[1:]))
	thisVector/=np.linalg.norm(thisVector)
	Vectors[thisLine[0]] = thisVector

pickle.dump(Vectors, sys.stdout)

import sys
import pickle
import numpy as np

Vectors = pickle.load(open(sys.argv[1],'r'))

Pairs = []

newVectors = []

for line in sys.stdin:
	(hyponym, hypernym) = line.split()
	Pairs.append((hyponym, hypernym))
	try:
		hypoVec = Vectors[hyponym]
		hyperVec = Vectors[hypernym]
		newVectors.append((np.concatenate((hypoVec,hyperVec),axis=1),(1,)))
		newVectors.append((np.concatenate((hyperVec,hypoVec),axis=1),(0,)))
	except KeyError:
		pass

pickle.dump(newVectors,sys.stdout)
	

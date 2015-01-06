import sys
import pickle
import numpy as np

Vectors = pickle.load(open(sys.argv[1],'r'))

Pairs = []

for line in sys.stdin:
	(hyponym, hypernym) = line.split()
	Pairs.append((hyponym, hypernym))
	
vectorShape = np.shape(Vectors[Pairs[1][0]])

Offset = np.zeros(vectorShape)

Count = 0

for (hyponym, hypernym) in Pairs:
	try:
		hyperVec = Vectors[hypernym]
		hypoVec = Vectors[hyponym]
		Offset = np.add(Offset,hyperVec)
		Offset = np.subtract(Offset,hypoVec)
		Count += 1
	except KeyError:
		pass

Offset = Offset/float(Count)

pickle.dump(Offset,sys.stdout)

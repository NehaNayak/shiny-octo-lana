import sys
import pickle
import numpy as np

Vectors = pickle.load(open(sys.argv[1],'r'))
Offset = pickle.load(open(sys.argv[2],'r'))

Pairs = []

for line in sys.stdin:
	(hyponym, hypernym) = line.split()
	Pairs.append((hyponym, hypernym))

for (hyponym, hypernym) in Pairs:
	try:
		hypoVec = Vectors[hyponym]
		candidate = np.add(hypoVec,Offset)
		Neighbours = sorted([(np.inner(candidate, vector),word) for word, vector in Vectors.iteritems()])
		Neighbours.reverse()
		Ranks = dict((key,i) for i,(cosine,key) in enumerate(Neighbours))
		print hyponym, hypernym, Ranks[hypernym]
		for i in Neighbours[:10]:
			print i

	except KeyError:
		pass

import sys
import pickle

hypoFile = open(sys.argv[2],'w')
hyperFile = open(sys.argv[3],'w')

Vectors = pickle.load(open(sys.argv[1],'r'))

pairs = []
for line in sys.stdin:
    (hypo, hyper) = line.split()
    if hypo in Vectors.keys() and hyper in Vectors.keys():
        hypoFile.write(hypo+"\t"+"\t".join(Vectors[hypo])+"\n")
        hyperFile.write(hyper+"\t"+"\t".join(Vectors[hyper])+"\n")


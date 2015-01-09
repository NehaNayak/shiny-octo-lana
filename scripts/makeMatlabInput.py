import sys

hypoFile = open(sys.argv[3],'w')
hyperFile = open(sys.argv[4],'w')

Vectors = pickle.load(open(sys.argv[1],'r'))

pairs = []
for line in sys.stdin:
    (hypo, hyper) = line.split()
    hypoFile.write("\t".join(Vectors[hypo])+"\n")
    hyperFile.write("\t".join(Vectors[hyper])+"\n")


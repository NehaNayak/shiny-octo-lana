import sys
import pickle

Vectors = {} 

for line in sys.stdin:
    thisLine = line.split()
    Vectors[thisLine[0]]=thisLine[1:]

pickle.dump(Vectors, sys.stdout)

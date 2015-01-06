import sys
import pickle
from collections import defaultdict

Parent = {} 
Ancestor = defaultdict(list)

def getParents(key,Parent, path):
    returnList = []
    path.append(key)
    if key in Parent.keys():
        parent = Parent[key]
        if parent not in path:
            returnList.append(parent)
            moreAncestors = getParents(parent,Parent,path)
            if moreAncestors is None:
                return None
            returnList+=moreAncestors
        else:
            return None
    return returnList
    
Words = set()

for line in sys.stdin:
    (w1, w2) = line.split()
    Parent[w2]=w1
    Words.add(w1)
    Words.add(w2)

for key in Words:
    Ancestor[key].append(key)
    ancestors = getParents(key,Parent,[])
    if ancestors is not None:
        Ancestor[key]+=ancestors

pickle.dump(dict(Ancestor),sys.stdout)

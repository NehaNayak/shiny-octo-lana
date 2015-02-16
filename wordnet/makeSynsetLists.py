import pickle
import nltk
import sys
from nltk.corpus import wordnet as wn

def printChildren(currSynset):
    if len(currSynset.hyponyms())==0:
        return
    else:
        if currSynset.lemmas()[0].name() in gloveVocab:
            synonyms = filter( lambda x : x in gloveVocab, map(lambda x:x.name(),currSynset.lemmas()))
            Synsets[currSynset.lemmas()[0].name()] = synonyms
        for hyponym in currSynset.hyponyms():
            printChildren(hyponym)

def main():
    global gloveVocab
    gloveVocab = []
    global Synsets
    Synsets = {}
    for line in open(sys.argv[1],'r'):
        gloveVocab.append(line[:-1])
    organism = wn.synsets("organism")[0]
    printChildren(organism)
    for word, synonyms in Synsets.iteritems():
        sys.stdout.write("-"+word+"\n"+"\n".join(synonyms)+"\n")
    

if __name__ =="__main__":
    main()
    

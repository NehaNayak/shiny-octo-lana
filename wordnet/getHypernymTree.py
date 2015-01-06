import nltk
import sys
from nltk.corpus import wordnet as wn

def printChildren(currSynset):
    if len(currSynset.hyponyms())==0:
        return
    else:
        for hyponym in currSynset.hyponyms():
            print currSynset.lemmas()[0].name(),hyponym.lemmas()[0].name()
        for hyponym in currSynset.hyponyms():
            printChildren(hyponym)

def main():
    organism = wn.synsets("organism")[0]
    printChildren(organism)
    

if __name__ =="__main__":
    main()
    

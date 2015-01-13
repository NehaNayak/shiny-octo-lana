import sys
import nltk
from nltk.stem import WordNetLemmatizer 

wnl = WordNetLemmatizer()

Vocab = []

for line in open(sys.argv[1],'r'):
    Vocab.append(line[:-1])

for line in sys.stdin:
    word = line[:-1]
    try:
        lemma = wnl.lemmatize(word,'n')
        if lemma is not word and lemma in Vocab:
            sys.stdout.write(lemma+"\t"+word+"\n")
    except UnicodeDecodeError:
        pass

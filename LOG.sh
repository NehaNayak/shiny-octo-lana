# Get word2vec vectors
cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.50.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_50.txt 
cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.200.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_200.txt 
cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.800.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_800.txt 

# Get word2vec vectors
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.50.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_50.txt 
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.200.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_200.txt 
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.800.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_800.txt 

#python wordnet/getHypernymTree.py > wordnet/organismHypernyms.txt
#cat wordnet/organismHypernyms.txt | python wordnet/makeAncestorDict.py
#cat wordnet/weighted_wordnet_vocabulary.txt | python wordnet/JCDistance.py wordnet/organismHypernyms_Ancestor.pickle > wordnet/organismHypernyms_JCD.txt

cat wordnet/organismHypernyms.txt |cut -f1,2 | awk '{print $1"\n"$2}' | sort | uniq > wordnet/organismHypernyms_vocab.txt
cat wordnet/organismHypernyms_JCD_limited.txt |cut -f1,2 | awk '{print $1"\n"$2}' | sort | uniq > wordnet/organismHypernyms_limited_vocab.txt

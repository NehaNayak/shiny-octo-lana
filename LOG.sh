# Get word2vec vectors
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.50.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_50.txt 
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.200.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_200.txt 
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.800.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/vocab.txt > data/SemEvalVectors_800.txt 

# numpyize vectors
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.50.txt | python scripts/numpyizeVectors.py > data/allVectors_50.pickle
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.200.txt | python scripts/numpyizeVectors.py > data/allVectors_200.pickle
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.800.txt | python scripts/numpyizeVectors.py > data/allVectors_800.pickle

#cat data/SemEvalVectors_50.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_50.pickle
#cat data/SemEvalVectors_200.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_200.pickle
#cat data/SemEvalVectors_800.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_800.pickle

# calculate offsets
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_50.pickle > data/Offset_ab_50.pickle
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_200.pickle > data/Offset_ab_200.pickle
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_800.pickle > data/Offset_ab_800.pickle

#cat SemEvalClassInclusion/cd_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_50.pickle > data/Offset_cd_50.pickle
#cat SemEvalClassInclusion/cd_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_200.pickle > data/Offset_cd_200.pickle
#cat SemEvalClassInclusion/cd_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_800.pickle > data/Offset_cd_800.pickle

# test offset
#cat SemEvalClassInclusion/ab_Test.txt| python scripts/testOffset.py data/allVectors_50.pickle data/Offset_ab_50.pickle > data/Offset_ab_50_results.txt
#cat SemEvalClassInclusion/ab_Test.txt| python scripts/testOffset.py data/allVectors_200.pickle data/Offset_ab_200.pickle > data/Offset_ab_200_results.txt
#cat SemEvalClassInclusion/ab_Test.txt| python scripts/testOffset.py data/allVectors_800.pickle data/Offset_ab_800.pickle > data/Offset_ab_800_results.txt

# test offset on train data
#head -n20 SemEvalClassInclusion/ab_Train.txt| python scripts/testOffset.py data/allVectors_50.pickle data/Offset_ab_50.pickle > data/Offset_ab_50_trainResults.txt
#head -n20 SemEvalClassInclusion/ab_Train.txt| python scripts/testOffset.py data/allVectors_200.pickle data/Offset_ab_200.pickle > data/Offset_ab_200_trainResults.txt
#head -n20 SemEvalClassInclusion/ab_Train.txt| python scripts/testOffset.py data/allVectors_800.pickle data/Offset_ab_800.pickle > data/Offset_ab_800_trainResults.txt

#plurals
#tr -cs 'a-z' '\n'< SemEvalClassInclusion/plurals.txt > SemEvalClassInclusion/plurals_vocab.txt
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.50.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/plurals_vocab.txt > data/SemEvalVectors_p_50.txt 
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.200.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/plurals_vocab.txt > data/SemEvalVectors_p_200.txt
#cat /scr/nlp/data/wordvectors/en/word2vec/wiki.bolt.giga5.f100.unk.neg5.800.txt | python scripts/getWord2VecVectors.py SemEvalClassInclusion/plurals_vocab.txt > data/SemEvalVectors_p_800.txt

# numpyize vectors
#cat data/SemEvalVectors_p_50.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_p_50.pickle
#cat data/SemEvalVectors_p_200.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_p_200.pickle
#cat data/SemEvalVectors_p_800.txt | python scripts/numpyizeVectors.py > data/SemEvalVectors_p_800.pickle

#sort -r SemEvalClassInclusion/plurals.txt > temp
#head -n80 temp > SemEvalClassInclusion/plurals_Train.txt
#tail -n56 temp > SemEvalClassInclusion/plurals_Test.txt

# train plurals
#cat SemEvalClassInclusion/plurals_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_p_50.pickle > data/Offset_plurals_50.pickle
#cat SemEvalClassInclusion/plurals_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_p_200.pickle > data/Offset_plurals_200.pickle
#cat SemEvalClassInclusion/plurals_Train.txt | python scripts/trainOffset.py data/SemEvalVectors_p_800.pickle > data/Offset_plurals_800.pickle

# test pliurals
#cat SemEvalClassInclusion/plurals_Test.txt| python scripts/testOffset.py data/allVectors_50.pickle data/Offset_plurals_50.pickle > data/Offset_plurals_50_results.txt
#cat SemEvalClassInclusion/plurals_Test.txt| python scripts/testOffset.py data/allVectors_200.pickle data/Offset_plurals_200.pickle > data/Offset_plurals_200_results.txt
#cat SemEvalClassInclusion/plurals_Test.txt| python scripts/testOffset.py data/allVectors_800.pickle data/Offset_plurals_800.pickle > data/Offset_plurals_800_results.txt

# pybrain stuff
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_50.pickle > data/pyBrainTrain_ab_50.pickle
#cat SemEvalClassInclusion/ab_Test.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_50.pickle > data/pyBrainTest_ab_50.pickle
cat SemEvalClassInclusion/ab_Train.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_200.pickle > data/pyBrainTrain_ab_200.pickle
cat SemEvalClassInclusion/ab_Test.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_200.pickle > data/pyBrainTest_ab_200.pickle
cat SemEvalClassInclusion/ab_Train.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_800.pickle > data/pyBrainTrain_ab_800.pickle
cat SemEvalClassInclusion/ab_Test.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_800.pickle > data/pyBrainTest_ab_800.pickle

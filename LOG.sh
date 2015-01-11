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
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_200.pickle > data/pyBrainTrain_ab_200.pickle
#cat SemEvalClassInclusion/ab_Test.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_200.pickle > data/pyBrainTest_ab_200.pickle
#cat SemEvalClassInclusion/ab_Train.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_800.pickle > data/pyBrainTrain_ab_800.pickle
#cat SemEvalClassInclusion/ab_Test.txt | python scripts/makePyBrainInput.py data/SemEvalVectors_800.pickle > data/pyBrainTest_ab_800.pickle
#python wordnet/getHypernymTree.py > wordnet/organismHypernyms.txt
#cat wordnet/organismHypernyms.txt | python wordnet/makeAncestorDict.py
#cat wordnet/weighted_wordnet_vocabulary.txt | python wordnet/JCDistance.py wordnet/organismHypernyms_Ancestor.pickle > wordnet/organismHypernyms_JCD.txt

#cat wordnet/organismHypernyms.txt |cut -f1,2 | awk '{print $1"\n"$2}' | sort | uniq > wordnet/organismHypernyms_vocab.txt
#cat wordnet/organismHypernyms_JCD_limited.txt |cut -f1,2 | awk '{print $1"\n"$2}' | sort | uniq > wordnet/organismHypernyms_limited_vocab.txt

#cat wordnet/organismHypernyms_JCD_limited.txt | python scripts/makeMatlabInput.py data/organismHypernyms_limited_vectors.pickle data/oHl_hypo.matrix data/oHl_hyper.matrix

#cat wordnet/organismHypernyms_JCD_limited.txt | python scripts/makeMatlabInput.py data/organismHypernyms_limited_vectors.pickle data/oHl_hypo.txt data/oHl_hyper.txt
#python scripts/makeTrainTestDev.py data/oHl_hypo.txt data/oHl_hyper.txt learnFunction1/oHl

for type in 'hypo' 'hyper'
do
    for tset in 'Train' 'Test' 'Dev'
    do
    echo 'learnFunction1/oHl_'$type'_'$tset'.txt'
    done
done

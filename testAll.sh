cut -f1 learnFunction1/data/oHl_hyper_Dev.txt > temp
for visibleSize in 50 #100 200 300
do
    for hiddenSize in 50 #100 500 1000 5000
    do
        inFileName='learnFunction1/data/hypers/hyper_v'$visibleSize'_h'$hiddenSize'_r0.01_l0.0001_b3.matrix' 
        outFileName='learnFunction1/data/hypers/hyper_v'$visibleSize'_h'$hiddenSize'_r0.01_l0.0001_b3.txt' 
        resultFileName='learnFunction1/data/results_'$visibleSize'_'$hiddenSize'.txt'
        gloveName='glove/glove.6B.'$visibleSize'd.pickle'
        paste temp $inFileName > $outFileName
        cat $outFileName | python test/nearestNeighbours_angle_choose.py $gloveName wordnet/organismHypernyms_limited_vocab.txt > $resultFileName
    done
done
rm temp

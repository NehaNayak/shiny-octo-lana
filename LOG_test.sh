for visibleSize in 50 #100 200 300
do
    for hiddenSize in 50 #100 500 1000 5000
    do
        inFileName='learnFunction1/data/hypers/hyper_train_v'$visibleSize'_h'$hiddenSize'_r0.01_l0.0001_b3.txt' 
        pickleFileName='glove/glove.6B.'$visibleSize'd.pickle'
        resultFileName='learnFunction1/data/results/results_train_v'$visibleSize'_h'$hiddenSize'_r0.01_l0.0001_b3.matrix' 
        cat $inFileName | python test/nearestNeighbours_angle_choose.py $pickleFileName > $resultFileName 
    done
done

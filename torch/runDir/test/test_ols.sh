for pairs in OH 
do
    for inputSize in 50 200
    do
        for iter in {500..10000..500}
        do
            echo -en 'ols\tw2v\t'$pairs'\t'$inputSize'\t'$iter'\t'
            devPairPath='../../pairFiles/glove'$pairs'_Dev.txt'
            trainPairPath='../../pairFiles/glove'$pairs'_Train.txt'
            modelPath='../params/ols_'$pairs'_model_in'$inputSize'_h500_lr0.5_il10000_vv_it'$iter'.th'
            torch-lua ../checks/justCheckLoss_syn.lua -inputSize $inputSize -pairPath $trainPairPath -modelPath $modelPath -synsetPath ../../wordnet/SynsetLists.th
            echo -en '\t'
            torch-lua ../checks/justCheckLoss_syn.lua -inputSize $inputSize -pairPath $devPairPath -modelPath $modelPath -synsetPath ../../wordnet/SynsetLists.th
            echo
        done
    done
done

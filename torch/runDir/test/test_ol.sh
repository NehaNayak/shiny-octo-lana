for pairs in OH OP OHM
do
    for inputSize in 100 200 300
    do
        for iter in {500..10000..500}
        do
            echo -en 'ol\tglove\t'$pairs'\t'$inputSize'\t'$iter'\t'
            devPairPath='../../pairFiles/glove'$pairs'_Dev.txt'
            trainPairPath='../../pairFiles/glove'$pairs'_Train.txt'
            modelPath='../params/ol_'$pairs'_model_in'$inputSize'_h500_lr0.5_il10000_vg_it'$iter'.th'
            torch-lua ../checks/justCheckLoss.lua -inputSize $inputSize -pairPath $trainPairPath -modelPath $modelPath -useGlove
            echo -en '\t'
            torch-lua ../checks/justCheckLoss.lua -inputSize $inputSize -pairPath $devPairPath -modelPath $modelPath -useGlove
            echo
        done
    done
done

for pairs in OH OP OHM
do
    for inputSize in 50 200
    do
        for iter in {500..10000..500}
        do
            echo -en 'ol\tw2v\t'$pairs'\t'$inputSize'\t'$iter'\t'
            devPairPath='../../pairFiles/glove'$pairs'_Dev.txt'
            trainPairPath='../../pairFiles/glove'$pairs'_Train.txt'
            modelPath='../params/ol_'$pairs'_model_in'$inputSize'_h500_lr0.5_il10000_vv_it'$iter'.th'
            torch-lua ../checks/justCheckLoss.lua -inputSize $inputSize -pairPath $trainPairPath -modelPath $modelPath
            echo -en '\t'
            torch-lua ../checks/justCheckLoss.lua -inputSize $inputSize -pairPath $devPairPath -modelPath $modelPath
            echo
        done
    done
done

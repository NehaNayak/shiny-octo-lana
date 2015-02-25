for pairs in OH OP OHM
do
    for inputSize in 100 200 300
    do
        echo -en 'ce\tglove\t'$pairs'\t'$inputSize'\t'$iter'\t'
        devPairPath='../../pairFiles/glove'$pairs'_Dev.txt'
        trainPairPath='../../pairFiles/glove'$pairs'_Train.txt'
        modelPath='../params/ce_'$pairs'_model_in'$inputSize'_vg.th'
        torch-lua ../checks/justCheckLoss_Diff.lua -inputSize $inputSize -pairPath $trainPairPath -modelPath $modelPath -useGlove
        echo -en '\t'
        torch-lua ../checks/justCheckLoss_Diff.lua -inputSize $inputSize -pairPath $devPairPath -modelPath $modelPath -useGlove
        echo
    done
done

for pairs in OH OP OHM
do
    for inputSize in 50 200
    do
        echo -en 'ce\tw2v\t'$pairs'\t'$inputSize'\t'$iter'\t'
        devPairPath='../../pairFiles/glove'$pairs'_Dev.txt'
        trainPairPath='../../pairFiles/glove'$pairs'_Train.txt'
        modelPath='../params/ce_'$pairs'_model_in'$inputSize'_vv.th'
        torch-lua ../checks/justCheckLoss_Diff.lua -inputSize $inputSize -pairPath $trainPairPath -modelPath $modelPath
        echo -en '\t'
        torch-lua ../checks/justCheckLoss_Diff.lua -inputSize $inputSize -pairPath $devPairPath -modelPath $modelPath
        echo
    done
done

cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
#for inputSize in 100 200 300
#do
#    for pairs in OH OP OHM
#    do
#        prefix='DiffCentrNorm'$pairs
#        pairFile='../pairFiles/glove'$pairs'_Train.txt'
#        torch-lua diffCentroidNorm.lua -inputSize $inputSize -prefix $prefix -outputDir params/ -pairPath $pairFile -useGlove
#    done
#done
#
#for inputSize in 50 200
#do
#    for pairs in OH OP OHM
#    do
#        prefix='DiffCentrNorm'$pairs
#        pairFile='../pairFiles/glove'$pairs'_Train.txt'
#        torch-lua diffCentroidNorm.lua -inputSize $inputSize -prefix $prefix -outputDir params/ -pairPath $pairFile
#    done
#done


#for inputSize in 100 200 300
#do
#    for pairs in OH OP OHM
#    do
#        prefix='DiffCentrNorm'$pairs
#        modelPath='params/DiffCentrNorm'$pairs'_model_in'$inputSize'_vg.th'
#        pairFile='../pairFiles/glove'$pairs'_Train.txt'
#
#        echo 'Glove '$pairs' Train '$inputSize
#        torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -useGlove -modelPath $modelPath | python results/sumCols.py
#
#        pairFile='../pairFiles/glove'$pairs'_Dev.txt'
#
#        echo 'Glove '$pairs' Dev '$inputSize
#        torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -useGlove -modelPath $modelPath | python results/sumCols.py
#    done
#done
#
#for inputSize in 50 200
#do
#    for pairs in OH OP OHM
#    do
#        prefix='DiffCentrNorm'$pairs
#        modelPath='params/DiffCentrNorm'$pairs'_model_in'$inputSize'_vv.th'
#        pairFile='../pairFiles/glove'$pairs'_Train.txt'
#
#        echo 'Word2Vec '$pairs' Train '$inputSize
#        torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -modelPath $modelPath | python results/sumCols.py
#
#        pairFile='../pairFiles/glove'$pairs'_Dev.txt'
#
#        echo 'Word2Vec '$pairs' Dev '$inputSize
#        torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -modelPath $modelPath | python results/sumCols.py
#    done
#done

pairs='OP'
for inputSize in 50 200
do
    prefix='DiffCentrNorm'$pairs
    modelPath='params/DiffCentrNorm'$pairs'_model_in'$inputSize'_vv.th'
    pairFile='../pairFiles/glove'$pairs'_Train.txt'

    echo 'Word2Vec '$pairs' Train '$inputSize
    torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -modelPath $modelPath | python results/sumCols.py

    pairFile='../pairFiles/glove'$pairs'_Dev.txt'

    echo 'Word2Vec '$pairs' Dev '$inputSize
    torch-lua ../figureOut/testDiff.lua -inputSize $inputSize -pairPath $pairFile -modelPath $modelPath | python results/sumCols.py

done

cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir

#fileName='gloveOH1_model_iterLimit_100_100_0.01'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_100_1000'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_100_100'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_100_500'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_200_1000'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 200 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_200_500'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 200 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOH1_model_iterLimit_300_500'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 300 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOP1_model_iterLimit_100_100'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOH_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOP_test_model_100_100'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOP_Dev.txt -modelPath $modelPath > $outputPath
#
#fileName='gloveOP1_model_iterLimit_100_100_0.01'
#modelPath='params/'$fileName'.th'
#outputPath='results/'$fileName'.txt'
#torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOP_Dev.txt -modelPath $modelPath > $outputPath

fileName='gloveOHM1_model_iterLimit_100_100_0.01'
modelPath='params/'$fileName'.th'
outputPath='results/'$fileName'.txt'
torch-lua ../figureOut/newTest.lua -inputSize 100 -pairPath ../pairFiles/gloveOHM_Dev.txt -modelPath $modelPath > $outputPath


cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayerNorm.lua -useGlove -inputSize 100 -hiddenSize 100 -prefix OL_OH_Norm -outputDir params/ -pairPath ../pairFiles/gloveOH_Train.txt

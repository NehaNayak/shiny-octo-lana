source ~/.bashrc
cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayerIterLimitLR.lua -inputSize 100 -hiddenSize 100 -learningRate 0.01 -prefix gloveOHM1_ -outputDir params/ -pairPath ../pairFiles/gloveOHM_Train.txt

source ~/.bashrc
cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayerIterLimitLR.lua -inputSize 200 -hiddenSize 5000 -learningRate 0.01 -prefix gloveOH1_ -outputDir params/ -pairPath ../pairFiles/gloveOH_Train.txt

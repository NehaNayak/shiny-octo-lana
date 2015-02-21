source ~/.bashrc
cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayerIterLimitW2v.lua -inputSize 200 -hiddenSize 200 -prefix w2vOP1_ -outputDir params/ -pairPath ../pairFiles/gloveOP_Train.txt

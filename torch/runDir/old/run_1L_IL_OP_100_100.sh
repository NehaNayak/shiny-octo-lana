source ~/.bashrc
cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayerIterLimit.lua -inputSize 100 -hiddenSize 100 -prefix gloveOP1_ -outputDir params/ -pairPath ../pairFiles/gloveOP_Train.txt

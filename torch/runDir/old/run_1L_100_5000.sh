source ~/.bashrc
cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
torch-lua oneLayer.lua -inputSize 100 -hiddenSize 5000 -prefix gloveOH1_ -outputDir params/ -pairPath ../pairFiles/gloveOH_Train.txt

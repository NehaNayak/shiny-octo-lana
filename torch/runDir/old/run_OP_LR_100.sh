cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
#torch-lua linReg.lua -inputSize 100 -prefix LinReg_OP -outputDir params/ -pairPath ../pairFiles/gloveOP_Train.txt
torch-lua linRegNormalized.lua -inputSize 100 -prefix LinRegNorm_OP -outputDir params/ -pairPath ../pairFiles/gloveOP_Train.txt

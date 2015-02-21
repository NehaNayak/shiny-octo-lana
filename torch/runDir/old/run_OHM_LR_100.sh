cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/runDir
#torch-lua linReg.lua -inputSize 100 -prefix LinReg_OHM -outputDir params/ -pairPath ../pairFiles/gloveOHM_Train.txt
torch-lua linRegNormalized.lua -inputSize 100 -prefix LinRegNorm_OHM -outputDir params/ -pairPath ../pairFiles/gloveOHM_Train.txt

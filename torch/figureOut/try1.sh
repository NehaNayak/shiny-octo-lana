source ~/.bashrc
cd ~/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut

for inputSize in 100
do
    for hiddenSize in 100 500 1000 5000
    do
        torch-lua try1.lua -inputSize $inputSize -hiddenSize $hiddenSize 
    done
done

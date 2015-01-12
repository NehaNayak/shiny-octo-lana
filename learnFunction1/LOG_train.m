visibleSize=50;
hiddenSize=50;
sparsityParam=0.01;
lambda=0.0001;
beta=3;
hypoTrainFileName='data/oHl_hypo_Train.matrix';
hyperTrainFileName='data/oHl_hyper_Train.matrix';

train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

%hiddenSize=100;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

%hiddenSize=500;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

%hiddenSize=1000;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

%hiddenSize=5000;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

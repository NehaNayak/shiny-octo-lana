visibleSize=100;
sparsityParam=0.01;
lambda=0.0001;
beta=3;
hypoTrainFileName='data/oHl_100_hypo_Train.matrix';
hyperTrainFileName='data/oHl_100_hyper_Train.matrix';

%hiddenSize=50;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

%hiddenSize=100;
%train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

hiddenSize=500;
train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

hiddenSize=1000;
train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

hiddenSize=5000;
train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

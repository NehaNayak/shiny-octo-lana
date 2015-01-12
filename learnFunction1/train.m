function train(visibleSize, hiddenSize, sparsity, lambda, beta)
visibleSize = 50; 
hiddenSize = 1000;     
sparsityParam = 0.01;   
lambda = 0.0001;     
beta = 3;            

hypoTrain = dlmread('data/oHl_hypo_Train.matrix')';
hyperTrain = dlmread('data/oHl_hyper_Train.matrix')';

theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 400;	  
options.display = 'on';

[opttheta, cost] = minFunc( @(p) costAndGrad(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, hypoTrain, hyperTrain), ...
                              theta, options);

dlmwrite('derp.txt',opttheta,'delimiter','\t')
end

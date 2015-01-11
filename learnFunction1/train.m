visibleSize = 50;   % number of input units 
hiddenSize = 1000;     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       

hypoTrain = dlmread('data/oHl_hypo_Train.matrix')';
hyperTrain = dlmread('data/oHl_hyper_Train.matrix')';

disp(size(hypoTrain))
disp(size(hyperTrain))

theta = initializeParameters(hiddenSize, visibleSize);
[cost, grad] = costAndGrad(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, hypoTrain, hyperTrain);

%checkNumericalGradient();

%numgrad = computeNumericalGradient( @(x) costAndGrad(x, visibleSize, ...
%                                                  hiddenSize, lambda, ...
%                                                  sparsityParam, beta, ...
%                                                  hypoTrain, hyperTrain), theta);

% Use this to visually compare the gradients side by side

% Compare numerically computed gradients with the ones obtained from backpropagation
%diff = norm(numgrad-grad)/norm(numgrad+grad);
%disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) costAndGrad(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, hypoTrain, hyperTrain), ...
                              theta, options);


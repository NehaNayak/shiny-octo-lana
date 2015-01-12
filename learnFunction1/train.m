function train(visibleSize, hiddenSize, sparsityParam, lambda, beta, thetaOutFileName, hypoTrainFileName, hyperTrainFileName)
	%visibleSize = 50;
	%hiddenSize = 1000;
	%sparsityParam = 0.01;
	%lambda = 0.0001;
	%beta = 3;

	hypoTrain = dlmread(hypoTrainFileName)';
	hyperTrain = dlmread(hyperTrainFileName)';

	theta = initializeParameters(hiddenSize, visibleSize);

	%  Use minFunc to minimize the function
	addpath minFunc/
	options.Method = 'lbfgs'; 
	%options.maxIter = 400;	  
	options.maxIter = 20;	  
	options.display = 'on';

	[opttheta, cost] = minFunc( @(p) costAndGrad(p, ...
		visibleSize, hiddenSize, ...
		lambda, sparsityParam, ...
		beta, hypoTrain, hyperTrain), ...
		theta, options);

	dlmwrite(thetaOutFileName,opttheta,'delimiter','\t')
end

function train(visibleSize, hiddenSize, sparsityParam, lambda, beta, thetaOutFileName, hypoTrainFileName, hyperTrainFileName)

	hypoTrain = dlmread(hypoTrainFileName)';
	hyperTrain = dlmread(hyperTrainFileName)';

	theta = initializeParameters(hiddenSize, visibleSize);

	%  Use minFunc to minimize the function
	addpath minFunc/
	options.Method = 'lbfgs'; 
	options.maxIter = 500;	  
	options.display = 'on';

	[opttheta, cost] = minFunc( @(p) costAndGrad(p, ...
		visibleSize, hiddenSize, ...
		lambda, sparsityParam, ...
		beta, hypoTrain, hyperTrain), ...
		theta, options);

	dlmwrite(thetaOutFileName,opttheta,'delimiter','\t')
end

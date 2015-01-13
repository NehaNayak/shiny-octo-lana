function train(visibleSize, hiddenSize, sparsityParam, lambda, beta, hypoTrainFileName, hyperTrainFileName)

    thetaOutFileName=strcat('data/thetas/theta_v',num2str(visibleSize),...
        '_h',num2str(hiddenSize),...
        '_r',num2str(sparsityParam),...
        '_l',num2str(lambda),...
        '_b',num2str(beta),'.matrix');

	hypoTrain = dlmread(hypoTrainFileName)';
	hyperTrain = dlmread(hyperTrainFileName)';

	theta = initializeParameters(hiddenSize, visibleSize);

	%  Use minFunc to minimize the function
	addpath minFunc/
	options.Method = 'lbfgs'; 
	options.maxIter = 1000;	  
	options.display = 'on';

	[opttheta, cost] = minFunc( @(p) costAndGrad(p, ...
		visibleSize, hiddenSize, ...
		lambda, sparsityParam, ...
		beta, hypoTrain, hyperTrain), ...
		theta, options);

	dlmwrite(thetaOutFileName,opttheta,'delimiter','\t')
end

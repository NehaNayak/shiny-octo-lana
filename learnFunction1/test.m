function test(thetaFileName, hypoTestFileName, hiddenSize, visibleSize)
    % unpack theta
    optTheta = dlmread(thetaFileName);
    [W1, W2, b1, b2] = unpackTheta(optTheta, hiddenSize, visibleSize); 

    % compute activations
    hypoTest= dlmread(hypoTestFileName)';
    [aInput, aHidden, aOutput] = computeActivations(hypoTest, W1, W2, b1, b2);
    % write calculated hypernym vectors to file
    hyperPredictFileName=strrep(thetaFileName,'theta','hyper');
    dlmwrite(hyperPredictFileName,aOutput','delimiter','\t');
end

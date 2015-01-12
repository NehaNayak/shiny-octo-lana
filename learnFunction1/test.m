function test(opttheta, hypoTest, hiddenSize, visibleSize)
    % unpack theta
    [W1, W2, b1, b2] = unpackTheta(opttheta, hiddenSize, visibleSize); 
    % compute activations
    [aInput, aHidden, aOutput] = computeActivations(hypoTest, W1, W2, b1, b2)
    % write calculated hypernym vectors to file
    dlmwrite('derp.txt',aOutput','delimiter','\t')
end

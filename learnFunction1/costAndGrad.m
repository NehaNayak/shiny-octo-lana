function [cost,grad] = costAndGrad(theta, visibleSize, hiddenSize, ...
                        lambda, sparsityParam, beta, hypoTrain, hyperTrain)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

[numFeatures, numSamples] = size(hypoTrain);

[aInput, zHidden, aHidden, zOutput, aOutput] = computeActivations(hypoTrain, W1, W2, b1, b2);

rho = sum(aHidden,2)./numSamples;

cost_squaredError = sum(sum(((aOutput-hyperTrain).^2)./2))/numSamples;
%cost_squaredError = sum(sum(((aOutput-hypoTrain).^2)./2))/numSamples;
cost_weightDecay = lambda*(sum(sum(W1.^2))+ sum(sum(W2.^2)))/2;
cost_sparsity = beta*sum(sparsityParam.*log(sparsityParam./rho) + (1-sparsityParam).*log((1-sparsityParam) ./(1-rho)));

cost = cost_squaredError + cost_weightDecay + cost_sparsity;
%cost = cost_squaredError + cost_weightDecay;
disp(cost)

%del_output = -(aInput-aOutput).*dSigmoid(zOutput);
del_output = -(hyperTrain-aOutput).*dSigmoid(zOutput);
del_hidden = bsxfun(@plus, (W2' * del_output), beta .* (-sparsityParam ./ rho + (1 - sparsityParam) ./ (1 - rho))).*dSigmoid(zHidden); 
%del_hidden = (W2' * del_output).*dSigmoid(zHidden); 
    
W1grad = (del_hidden*aInput')./numSamples + lambda.*W1;
b1grad = sum(del_hidden,2)./numSamples;
W2grad = (del_output*aHidden')./numSamples+lambda.*W2;
b2grad = sum(del_output,2)./numSamples;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end


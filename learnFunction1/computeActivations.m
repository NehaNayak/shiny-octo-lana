%-------------------------------------------------------------------
% Calculate activations

%function [aInput, zHidden, aHidden, zOutput, aOutput] = activations(data, W1, W2, b1, b2)
function [aInput, aHidden, aOutput] = activations(data, W1, W2, b1, b2)
    aInput = data;
    zHidden = bsxfun(@plus,W1*aInput,b1);
    aHidden = sigmoid(zHidden);
    zOutput = bsxfun(@plus,W2*aHidden,b2);
    aOutput = sigmoid(zOutput);
end 

%-------------------------------------------------------------------
% Calculate dSigmoid

function dSigm = dSigmoid(x)
    dSigm = sigmoid(x).*(1-sigmoid(x));
end
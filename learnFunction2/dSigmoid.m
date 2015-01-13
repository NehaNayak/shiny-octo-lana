%-------------------------------------------------------------------
% Calculate dSigmoid

function dSigm = dSigmoid(x)
%    dSigm = sigmoid(x).*(1-sigmoid(x));
    dSigm = x.*(1.0-x);
end

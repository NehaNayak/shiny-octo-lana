require('torch')
print('i')
cmd = torch.CmdLine()
cmd:option('-inputSize',50,'size of input layer')
cmd:option('-hiddenSize',100,'size of hidden layer')
params = cmd:parse(arg)
print(params)
for k,v in pairs(params) do
    print(k)
    print(v)
    print(type(v))
end

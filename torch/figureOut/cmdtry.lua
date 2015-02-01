require('torch')

cmd = torch.CmdLine()
cmd:option('-inputSize',50,'size of input layer')
cmd:option('-hiddenSize',100,'size of hidden layer')
cmd:option('-outputDir','./','size of hidden layer')
cmdparams = cmd:parse(arg)

local output_path = outputDir .. '_' .. inputSize .. '_' .. hiddenSize .. '.th'
print(output_path)
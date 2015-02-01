require('torch')
require('unsup')
require('torchnlp')

cmd = torch.CmdLine()
cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',50,'size of hidden layer')
cmd:option('-outputDir','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmdparams = cmd:parse(arg)

local decoder_output_path = cmdparams.outputDir .. 'decoder_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'
local encoder_output_path = cmdparams.outputDir .. 'encoder_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'

print(encoder_output_path)

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize .. 'd.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

local m = torch.randn(cmdparams.inputSize)
dataset = nil
local f =assert(io.open("temp.txt", "r"))
while true do
  line = f:read()
  if not line then break end
  for win,wout in string.gmatch(line, "(%w+)%s(%w+)") do
    local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
    local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
    if dataset==nil then
        dataset = vin:clone()
    else
        dataset = torch.cat(dataset,vin,2)
    end
  end
end

params = {}
params.beta = 1
params.maxiter = 1000000
params.batchsize = 1
params.statinterval = 5000
params.eta = 2e-3
params.etadecay = 1e-5
params.momentum = 0

--------------------------------------------------------------------
-- create model
--
inputSize = cmdparams.inputSize -- input size

model = nn.Sequential()
modelLinear = nn.Linear(inputSize,inputSize)
model:add(modelLinear)
criterion = nn.MSECriterion()

--------------------------------------------------------------------
-- trainable parameters
--   
--
-- get all parameters
input = torch.randn(inputSize)
output = model:forward(input)
grad_wrt_output = torch.randn(inputSize)
grad_wrt_input = model:backward(input, grad_wrt_output)
x, dl_dx = model:getParameters()  

feval = function(x_new)
    --if x ~= x_new then
    --    x:copy(x_new)
    --end

    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > (#dataset)[1] then _nidx_ = 1 end

    local sample = dataset[_nidx_]
    local target = sample[{1}]
    local inputs = sample[{1}]

    dl_dx:zero()

    local loss_x = criterion:forward(model:forward(inputs),target)
    model:backward(inputs, criterion:backward(model.output,target))

    return loss_x, dl_dx
end

sgd_params = {
    learningRate = 1e-3,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}

-- SGD
--
for i = 1,10000 do
    current_loss = 0

    for i = 1,5000 do
        _, fs = optim.sgd(feval,x,sgd_params)
        current_loss = current_loss + fs[1]
    end

    current_loss = current_loss/ (#dataset)
    print('current loss = ' .. current_loss)
end

require('torch')
require('unsup')
require('torchnlp')

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

cmd = torch.CmdLine()
cmd:option('-inputSize',50,'size of input layer')
cmd:option('-hiddenSize',100,'size of hidden layer')
cmd:option('-outputDir','./','size of hidden layer')
cmdparams = cmd:parse(arg)

local decoder_output_path = outputDir .. 'decoder_' .. inputSize .. '_' .. hiddenSize .. '.th'
local encoder_output_path = outputDir .. 'encoder_' .. inputSize .. '_' .. hiddenSize .. '.th'

print(encoder_output_path)

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize .. 'd.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

params = {}
params.maxiter=1000000
params.statinterval=5000
params.batchsize=1

local m = torch.randn(300)
dataset = {}

local f =assert(io.open("temp.txt", "r"))
while true do
  line = f:read()
  if not line then break end
  for win,wout in string.gmatch(line, "(%w+)%s(%w+)") do
    local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
    local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
    table.insert(dataset,{vin,vout})
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
inputSize = 300 -- input size
outputSize = 1000 -- output size

-- encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))
decoder:add(nn.Tanh())

-- complete model
module = unsup.AutoEncoder(encoder, decoder, params.beta)

--------------------------------------------------------------------
-- trainable parameters
--   
--
-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()  

--------------------------------------------------------------------
-- train model
--

print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do
    --------------------------------------------------------------------
    -- progress
    --
    iter = iter+1
    xlua.progress(iter, params.statinterval)
    --------------------------------------------------------------------
    -- create mini-batch
    --
    local example = dataset[t]
    local inputs = {}
    local targets = {}
    for i = t,t+params.batchsize-1 do
        -- load new sample
        local input = dataset[i%5000+1][1]:clone()
        local target = dataset[i%5000+1][2]:clone()
        table.insert(inputs, input)
        table.insert(targets, target)
    end
   --------------------------------------------------------------------
   -- define eval closure
   --

   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      for i = 1,#inputs do
         -- f
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
      end

      -- normalize
      dl_dx:div(#inputs)
      f = f/#inputs

      -- return f and df/dx
      return f,dl_dx
   end

--------------------------------------------------------------------
   -- one SGD step
   --
   sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]

   --------------------------------------------------------------------
   -- compute statistics / report error
   --
   if math.fmod(t , params.statinterval) == 0 then

      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)

      -- reset counters
      err = 0; iter = 0
   end
end

print(encoder.weight)
print(encoder.bias)
print(decoder.weight)
print(decoder.bias)

torch.save(encoder_output_path, encoder)
torch.save(decoder_output_path, decoder)

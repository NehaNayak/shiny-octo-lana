require('torch')
require('unsup')
require('torchnlp')

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

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
--params.maxiter = 5000
params.batchsize = 1
params.statinterval = 5000
params.eta = 2e-3
params.etadecay = 1e-5
params.momentum = 0

--------------------------------------------------------------------
-- create model
--
inputSize = cmdparams.inputSize -- input size
outputSize = cmdparams.hiddenSize -- output size

-- encoder
model = nn.Sequential()
encoderLinear = nn.Linear(inputSize,outputSize)
model:add(encoderLinear)
model:add(nn.Tanh())

-- decoder
decoderLinear = nn.Linear(outputSize,inputSize)
model:add(decoderLinear)
model:add(nn.Tanh())

--------------------------------------------------------------------
-- trainable parameters
--   
--
-- get all parameters
x,dl_dx,ddl_ddx = model:getParameters()  

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
         f = f + model:updateOutput(inputs[i], targets[i])

         -- gradients
         model:updateGradInput(inputs[i], targets[i])
         model:accGradParameters(inputs[i], targets[i])
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

torch.save(encoder_output_path, encoderLinear)
torch.save(decoder_output_path, decoderLinear)

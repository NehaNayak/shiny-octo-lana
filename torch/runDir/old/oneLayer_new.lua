require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

----------------------------------------------------------------------
-- Hyperparams

cmd = torch.CmdLine()
cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',500,'size of hidden layer')
cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')
cmdparams = cmd:parse(arg)

local output_path = cmdparams.outputDir .. cmdparams.prefix .. 'model_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'

----------------------------------------------------------------------
-- Vectors

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

----------------------------------------------------------------------
-- Training data 

local m = torch.randn(cmdparams.inputSize)
local f =assert(io.open(cmdparams.pairPath, "r"))
while true do
  line = f:read()
  if not line then break end
  for wout,win in string.gmatch(line, "(%w+)%s(%w+)") do
    local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
    local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
    if dataset_in==nil then
        dataset_in = vin:clone()
        dataset_out = vout:clone()
    else
        dataset_in = torch.cat(dataset_in,vin,2)
        dataset_out = torch.cat(dataset_out,vout,2)
    end
  end
end

dataset_in = dataset_in:t()
dataset_out = dataset_out:t()

----------------------------------------------------------------------
-- Model

model = nn.Sequential() 
model:add(nn.Linear(cmdparams.inputSize, cmdparams.hiddenSize))
model:add(nn.Tanh())
model:add(nn.Linear(cmdparams.hiddenSize, cmdparams.inputSize))
model:add(nn.Tanh())
criterion = nn.CosineEmbeddingCriterion()

----------------------------------------------------------------------
-- Train

x, dl_dx = model:getParameters()

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#dataset_in)[1] then _nidx_ = 1 end

   local input_sample = dataset_in[_nidx_]
   local target_sample = dataset_out[_nidx_]

   local target = target_sample:clone()
   local inputs = input_sample:clone()

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local a = model:forward(inputs)
   local b = target
   print(a)
   print(b)

   local loss_x = criterion:forward({a,b},1)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-correlation.

-- we cycle 1e4 times over our training data
for i = 1,1e5 do

   -- this variable is used to estimate the average loss
   prev_loss = current_loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#dataset_in)[1] do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _,fs = optim.adagrad(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#dataset_in)[1]
   print('current loss = ' .. current_loss)

   if prev_loss~= nil then
      if (prev_loss - current_loss)/current_loss < 10e-6 then
         break
      end
   end

end

torch.save(output_path, model)
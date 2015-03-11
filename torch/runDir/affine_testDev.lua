require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')

cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','./','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')

cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-iterLimit',10e3,'maximum number of iterations')

cmd:option('-useGlove',false,'whether to use Glove or word2vec')

cmdparams = cmd:parse(arg)

if cmdparams.useGlove then
    vecset = 'g'
else
    vecset = 'v'
end

local output_path = table.concat({
				cmdparams.outputDir, 
				cmdparams.prefix,
				'_model',
				'_in',
				cmdparams.inputSize,
				'_lr',
				cmdparams.learningRate,
				'_il',
				cmdparams.iterLimit,
                '_v',
                vecset,
				},"")

-- Load word embeddings

if cmdparams.useGlove then
	emb_dir = '/scr/kst/data/wordvecs/glove/'
	emb_prefix = emb_dir .. 'glove.6B'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
else
    emb_dir = '/scr/kst/data/wordvecs/word2vec/'
    emb_prefix = emb_dir .. 'wiki.bolt.giga5.f100.unk.neg5'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'.th')
end

-- Create dataset

local m = torch.randn(cmdparams.inputSize):zero()
local f =assert(io.open(cmdparams.pairPath, "r"))
while true do
  line = f:read()
  if not line then break end
  for wout,win in string.gmatch(line, "(%w+)%s(%w+)") do
    local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
    local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
    if dataset_in==nil then
        dataset_in = vin:clone()/vin:norm()
        dataset_out = vout:clone()/vout:norm()
    else
        dataset_in = torch.cat(dataset_in,vin/vin:norm(),2)
        dataset_out = torch.cat(dataset_out,vout/vout:norm(),2)
    end
  end
end

dataset_in = dataset_in:t()
dataset_out = dataset_out:t()

-- Define model

model = nn.Sequential() -- define the container
model:add(nn.Linear(cmdparams.inputSize, cmdparams.inputSize)) -- define the only module

criterion = nn.MSECriterion()

-- Train

x, dl_dx = model:getParameters()

-- Define closure

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
   local loss_x = criterion:forward(model:forward(inputs), target)
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
   learningRate = cmdparams.learningRate,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- we cycle 1e4 times over our training data
for i = 1,cmdparams.iterLimit do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#dataset_in)[1] do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#dataset_in)[1]
   if i % 500 == 0 then
      torch.save(output_path .. '_it' .. i .. '.th' , model)
   end
   print('current loss = ' .. current_loss)

end

torch.save(output_path .. '.th' , model)

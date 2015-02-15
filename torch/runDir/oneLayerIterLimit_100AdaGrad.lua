----------------------------------------------------------------------
-- example-linear-regression.lua
-- 
-- This script provides a very simple step-by-step example of
-- linear regression, using Torch7's neural network (nn) package,
-- and the optimization package (optim).
--

-- note: to run this script, simply do:
-- torch script.lua

-- to run the script, and get an interactive shell once it terminates:
-- torch -i script.lua

-- we first require the necessary packages.
-- note: optim is a 3rd-party package, and needs to be installed
-- separately. This can be easily done using Torch7's package manager:
-- torch-pkg install optim

require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

----------------------------------------------------------------------
-- 1. Create the training data

-- In all regression problems, some training data needs to be 
-- provided. In a realistic scenarios, data comes from some database
-- or file system, and needs to be loaded from disk. In that 
-- tutorial, we create the data source as a Lua table.

-- In general, the data can be stored in arbitrary forms, and using
-- Lua's flexible table data structure is usually a good idea. 
-- Here we store the data as a Torch Tensor (2D Array), where each
-- row represents a training sample, and each column a variable. The
-- first column is the target variable, and the others are the
-- input variables.

-- The data are from an example in Schaum's Outline:
-- Dominick Salvator and Derrick Reagle
-- Shaum's Outline of Theory and Problems of Statistics and Economics
-- 2nd edition
-- McGraw-Hill
-- 2002

-- The data relate the amount of corn produced, given certain amounts
-- of fertilizer and insecticide. See p 157 of the text.

-- In this example, we want to be able to predict the amount of
-- corn produced, given the amount of fertilizer and intesticide used.
-- In other words: fertilizer & insecticide are our two input variables,
-- and corn is our target value.

--  {corn, fertilizer, insecticide}

cmd = torch.CmdLine()
cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',500,'size of hidden layer')
cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')
cmdparams = cmd:parse(arg)

local output_path = cmdparams.outputDir .. cmdparams.prefix .. 'model_iterLimit_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

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
-- 2. Define the model (predictor)

-- The model will have one layer (called a module), which takes the 
-- 2 inputs (fertilizer and insecticide) and produces the 1 output 
-- (corn).

-- Note that the Linear model specified below has 3 parameters:
--   1 for the weight assigned to fertilizer
--   1 for the weight assigned to insecticide
--   1 for the weight assigned to the bias term

-- In some other model specification schemes, one needs to augment the
-- training data to include a constant value of 1, but this isn't done
-- with the linear model.

-- The linear model must be held in a container. A sequential container
-- is appropriate since the outputs of each module become the inputs of 
-- the subsequent module in the model. In this case, there is only one
-- module. In more complex cases, multiple modules can be stacked using
-- the sequential container.

-- The modules are all defined in the neural network package, which is
-- named 'nn'.

model = nn.Sequential()                 -- define the container
model:add(nn.Linear(cmdparams.inputSize, cmdparams.hiddenSize)) -- define the only module
model:add(nn.Tanh())
model:add(nn.Linear(cmdparams.hiddenSize, cmdparams.inputSize))
model:add(nn.Tanh())

----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

criterion = nn.MSECriterion()


----------------------------------------------------------------------
-- 4. Train the model

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:

x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our mode, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   -- if _nidx_ > (#data)[1] then _nidx_ = 1 end
   if _nidx_ > (#dataset_in)[1] then _nidx_ = 1 end

   --local sample = data[_nidx_]

   local input_sample = dataset_in[_nidx_]
   local target_sample = dataset_out[_nidx_]
   --local target = sample[{ {1} }]      -- this funny looking syntax allows
   --local inputs = sample[{ {2,3} }]    -- slicing of arrays.
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
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-correlation.

-- we cycle 1e4 times over our training data
for i = 1,1e4 do

   -- this variable is used to estimate the average loss
   prev_loss = current_loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   -- for i = 1,(#data)[1] do
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

   --if prev_loss~= nil then
   --   if (prev_loss - current_loss)/current_loss < 10e-5 then
   --      break
   --   end
   --end

end

torch.save(output_path, model)

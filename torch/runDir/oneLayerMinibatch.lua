require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',500,'size of hidden layer')

cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','./','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')

cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-iterLimit',10e4,'max number of iterations')
cmd:option('-batchSize',10,'minibatch size')

cmd:option('-useGlove',false,'use glove or word2vec')

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
				'_h',
				cmdparams.hiddenSize,
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

model = nn.Sequential()                 
model:add(nn.Linear(cmdparams.inputSize, cmdparams.hiddenSize)) 
model:add(nn.Tanh())
model:add(nn.Linear(cmdparams.hiddenSize, cmdparams.inputSize))
model:add(nn.Tanh())

criterion = nn.CosineEmbeddingCriterion()

-- Train

x, dl_dx = model:getParameters()

-- SGD

sgd_params = {
   learningRate = cmdparams.learningRate,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- we cycle 1e4 times over our training data
for i = 1, cmdparams.iterLimit, cmdparams.batchSize do

   --------------------------------------------------------------------
   -- create mini-batch
   --
   local inputs = {}
   local targets = {}
   for t = i,i+cmdparams.batchSize-1 do
      -- load new sample
      index = t%((#dataset_in)[1])+1
      local input = dataset_in[index]:clone()
      local target = dataset_out[index]:clone()
      table.insert(inputs, input)
      table.insert(targets, target)
   end

   --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      local f = 0
      dl_dx:zero()
      for j = 1,#inputs do
         f = f + criterion:forward({model:forward(inputs[j]), targets[j]},1)
         -- gradients
         model:backward(inputs[j], criterion:backward({model.output, targets[j]},1)[1])
       --f = f + criterion:forward({model:forward(inputs), targets},1)
       --model:backward(inputs, criterion:backward({model.output, targets},1))
      end
      -- normalize
      dl_dx:div(#inputs)
      f = f/#inputs
      -- return f and df/dx
      return f,dl_dx
   end

   -- this variable is used to estimate the average loss
   current_loss = 0
   _,fs = optim.sgd(feval,x,sgd_params)
   current_loss = current_loss + fs[1]

   -- report average error on epoch
   current_loss = current_loss / (#dataset_in)[1]
   if i % 500 == 0 then
      torch.save(output_path .. '_it' .. i .. '.th' , model)
   end
   print('current loss = ' .. current_loss)

end

torch.save(output_path .. '.th', model)

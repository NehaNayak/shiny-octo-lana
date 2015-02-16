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

cmd:option('-learningRate',0.01,'learning rate')
cmd:option('-iterLimit',10e4,'maximum number of iterations')

cmd:option('-useGlove',true,'whether to use Glove or word2vec')

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
                vecset,
				},"")

-- Load word embeddings

if cmdparams.useGlove then
	emb_dir = '/scr/kst/data/wordvecs/glove/'
	emb_prefix = emb_dir .. 'glove.6B'
else
    emb_dir = '/scr/kst/data/wordvecs/word2vec/'
    emb_prefix = emb_dir .. 'wiki.bolt.giga5.f100.unk.neg5'
end
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')

-- Create dataset

local m = torch.randn(cmdparams.inputSize):zero()
in_centroid = m:clone()
out_centroid = m:clone()
datasetSize = 0
local f =assert(io.open(cmdparams.pairPath, "r"))
while true do
  line = f:read()
  if not line then break end
  for wout,win in string.gmatch(line, "(%w+)%s(%w+)") do
    local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
    local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
    in_centroid = in_centroid + vin
    out_centroid = out_centroid + vout
    datasetSize = datasetSize + 1
  end
end

diffCentroid = (out_centroid - in_centroid)/datasetSize

-- Define model

model = nn.Sequential() -- define the container
linearLayer = nn.Linear(cmdparams.inputSize, cmdparams.inputSize)
linearLayer.weight:zero()
linearLayer.bias = diffCentroid
model:add(linearLayer) -- define the only module

torch.save(output_path .. '.th' , diffCentroid)

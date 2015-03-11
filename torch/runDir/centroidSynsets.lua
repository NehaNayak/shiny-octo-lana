require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

function readSet(size,path)
    local m = torch.randn(size):zero()
    local f =assert(io.open(path, "r"))

    while true do
        line = f:read()
        if not line then break end
        for wout,win in string.gmatch(line, "(%w+)%s(%w+)") do
            local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
            local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
            if set_in ==nil then
                set_in = vin:clone()/vin:norm()
                set_out= vout:clone()/vout:norm()
            else
                set_in = torch.cat(set_in,vin/vin:norm(),2)
                set_out = torch.cat(set_out,vout/vout:norm(),2)
            end
        end
    end
    
    return {set_in:t(),set_out:t()}
   
end

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')

cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','./','where to put serialized params')
cmd:option('-trainPath','../pairFiles','where to find word pairs')
cmd:option('-devPath','../pairFiles','where to find word pairs')


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
    in_centroid = in_centroid + vin/vin:norm()
    out_centroid = out_centroid + vout/vout:norm()
    datasetSize = datasetSize + 1
  end
end

diffCentroid = (out_centroid - in_centroid)/datasetSize

torch.save(output_path .. '.th' , diffCentroid)

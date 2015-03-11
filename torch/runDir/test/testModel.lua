require 'nn'
require 'torch'
require 'torchnlp'

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

cmd = torch.CmdLine()
cmd:option('-inputSize',100, 'input vector size')
cmd:option('-modelPath','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')
cmd:option('-useGlove',false,'use glove or word2vec')
cmdparams = cmd:parse(arg)

model = torch.load(cmdparams.modelPath)

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

for i = 1,(#dataset_in)[1] do
   local myPrediction = model:forward(dataset_in[i])
   predictedCosine = cosine(myPrediction,dataset_out[i])
   selfCosine = cosine(dataset_in[i],dataset_out[i])

   cosines = {}
   for j = 1,20000 do
      table.insert(cosines,cosine(dataset_out[i], emb_vecs[j]:typeAs(m)))
   end
   table.insert(cosines,predictedCosine)
   table.insert(cosines,selfCosine)
   table.sort(cosines)
   local predRank
   local selfRank
   for i,n in ipairs(cosines) do
      if n==predictedCosine then
         predRank=20002-i
      end
      if n==selfCosine then
         selfRank=20002-i
      end
   end
   if predRank ~= nil and selfRank ~= nil then
      print("Pred\t" .. predRank .. "\tSelf\t" .. selfRank)
   end
end

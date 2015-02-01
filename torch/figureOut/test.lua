function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

cmd = torch.CmdLine()
cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',500,'size of hidden layer')
cmd:option('-prefix','_','size of input layer')
cmd:option('-modelPath','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmd:option('-pairPath','../pairFiles','where to find word pairs')
cmdparams = cmd:parse(arg)

local model_path = cmdparams.modelPath .. 'model_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'
model = torch.load(model_path)

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
    print(win)
    if dataset_in==nil then
        dataset_in = vin:clone()
        dataset_out = vout:clone()
    else
        dataset_in = torch.cat(dataset_in,vin,2)
      dataset_out = torch.cat(dataset_out,vin,2)
    end
  end
end

dataset_in = dataset_in:t()
dataset_out = dataset_out:t()

for i = 1,(#dataset_in)[1] do
   local myPrediction = model:forward(dataset_in[i])
   print(cosine(dataset_in[i],dataset_out[i]) .. '\t' .. cosine(myPrediction,dataset_out[i]))
end



require('torch')
require('unsup')
require('torchnlp')

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

function apply(input, encoderLinear, decoderLinear)
    return (input*encoderLinear.weight+encoderLinear.bias)*decoderLinear.weight + decoderLinear.bias
end

cmd = torch.CmdLine()
cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',50,'size of hidden layer')
cmd:option('-inputDir','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to find serialized params')
cmdparams = cmd:parse(arg)

local decoder_input_path = cmdparams.inputDir .. 'decoder_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'
local encoder_input_path = cmdparams.inputDir .. 'encoder_' .. cmdparams.inputSize .. '_' .. cmdparams.hiddenSize .. '.th'

encoderLinear = torch.load(encoder_input_path)
decoderLinear = torch.load(decoder_input_path)

print(encoderLinear)
print(decoderLinear)

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

-- encoder
encoder = nn.Sequential()
encoder:add(encoderLinear)
encoder:add(nn.Tanh())
encoder:add(nn.Diag(cmdparams.hiddenSize))

-- decoder
decoder = nn.Sequential()
decoder:add(decoderLinear)
decoder:add(nn.Tanh())

module = unsup.AutoEncoder(encoder, decoder, 1)

for k, vecs in pairs(dataset) do
    vin = vecs[1]
    vout = vecs[2]
    vout_ = module:forward(vecs)
    print(vout_)
    print(cosine(vin,vout))
    print(cosine(vout_,vout))
end


require('torch')
require('unsup')
require('torchnlp')

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.300d.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

params = {}
params.maxiter=1000000
params.statinterval=5000
params.batchsize=1


local a = torch.randn(300)
local in1 = 'king'
local in2 = 'actor'
local out1 = 'queen'
local out2 = 'actress'
local vin1 = emb_vecs[emb_vocab:index(in1)]:typeAs(a)
local vin2 = emb_vecs[emb_vocab:index(in2)]:typeAs(a)
local vout1 = emb_vecs[emb_vocab:index(out1)]:typeAs(a)
local vout2 = emb_vecs[emb_vocab:index(out2)]:typeAs(a)
local bothIn = torch.cat(vin1,vin2,2):t()
local bothOut = torch.cat(vout1,vout2,2):t()

dataset = {}
dataset[1] = {vin1, vout1}
dataset[2] = {vin2, vout2}

params = {}
params.beta = 1
params.maxiter = 1000000
params.batchsize = 1
params.statinterval = 5000
params.eta = 2e-3
params.etadecay = 1e-5
params.momentum = 0

--------------------------------------------------------------------
-- create model
--
inputSize = 300 -- input size
outputSize = 1000 -- output size

-- encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))
decoder:add(nn.Tanh())

-- complete model
module = unsup.AutoEncoder(encoder, decoder, params.beta)

--------------------------------------------------------------------
-- trainable parameters
--   
--
-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()  

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
        print(i) 
        -- load new sample
        local input = dataset[math.mod(i,2)+1][1]:clone()
        local target = dataset[math.mod(i,2)+1][2]:clone()
        --local input = dataset[i][1]:clone()
        --local target = dataset[i][2]:clone()
        -- local input = sample[1]:clone()
        -- local input = x:clone()
        --local input = clone(sample[1])
        -- local target = sample[2]:clone()
        table.insert(inputs, input)
        table.insert(targets, target)
    end

end

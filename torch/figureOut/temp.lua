require('torchnlp')

print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.300d.th')
print('vocab size = ' .. emb_vecs:size(1))
print('dimension = ' .. emb_vecs:size(2))

local m = torch.randn(300)
dataset = {}

local f =assert(io.open("temp.txt", "r"))
words = {}
while true do
  line = f:read()
  if not line then break end
  t = {}
  for k,v in string.gmatch(line, "(%w+)%s(%w+)") do
    print(k)
    print(v)
    local vin1 = emb_vecs[emb_vocab:index(k)]:typeAs(m)
    local vin2 = emb_vecs[emb_vocab:index(v)]:typeAs(m)
    table.insert(dataset,{vin1,vin2})
  end
  table.insert(words,line)
end


require('torch')
require('unsup')
require('torchnlp')

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

--print('loading word embeddings')
local emb_dir = '/scr/kst/data/wordvecs/glove/'
local emb_prefix = emb_dir .. 'glove.6B'
local emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.300d.th')
---print('vocab size = ' .. emb_vecs:size(1))
--print('dimension = ' .. emb_vecs:size(2))

--print(emb_vecs._tokens)

for i=1,emb_vecs:size(1) do
print(emb_vocab:token(i))
end


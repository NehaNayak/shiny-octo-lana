require 'nn'
require 'torch'
require 'torchnlp'

function cosine(v, w)
  return v:dot(w) / v:norm() / w:norm()
end

function readSet(size,pairPath,synsetPath)
    
    Synsets = torch.load(synsetPath)

    local m = torch.randn(size):zero()
    local f =assert(io.open(pairPath, "r"))

    while true do
        line = f:read()
        if not line then break end
        for wout,win in string.gmatch(line, "(%w+)%s(%w+)") do
            local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
            local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
            
            if Synsets[win]==nil then
                vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
            else
                vin = m:clone()
                for i, word in ipairs(Synsets[win]) do
                    vin = vin + emb_vecs[emb_vocab:index(word)]:typeAs(m)
                end
            end
            if Synsets[wout]==nil then
                vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
            else
                vout = m:clone()
                for i, word in ipairs(Synsets[wout]) do
                    vout = vout + emb_vecs[emb_vocab:index(word)]:typeAs(m)
                end
            end

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

cmd = torch.CmdLine()
cmd:option('-inputSize',100, 'input vector size')
cmd:option('-modelPath','/afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/shiny-octo-lana-2/shiny-octo-lana/torch/figureOut/params/','where to put serialized params')
cmd:option('-trainPath','../pairFiles','where to find word pairs')
cmd:option('-devPath','../pairFiles','where to find word pairs')
cmd:option('-synsetPath','../../SynsetLists.th','where to find synsets')
cmd:option('-useGlove',false,'whether to use Glove or word2vec')

cmdparams = cmd:parse(arg)

if cmdparams.useGlove then
    vecset = 'g'
else
    vecset = 'v'
end

offset = torch.load(cmdparams.modelPath)

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

dataset = readSet(cmdparams.inputSize, cmdparams.trainPath,cmdparams.synsetPath) 
dataset_in=dataset[1]
dataset_out=dataset[2]

for i = 1,(#dataset_in)[1] do
   local myPrediction = dataset_in[i]+offset
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

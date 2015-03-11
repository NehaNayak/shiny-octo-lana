require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

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
                    vin:cmul(emb_vecs[emb_vocab:index(word)]:typeAs(m))
                end
            end
            if Synsets[wout]==nil then
                vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
            else
                vout = m:clone()
                for i, word in ipairs(Synsets[wout]) do
                    vout:cmul(emb_vecs[emb_vocab:index(word)]:typeAs(m))
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

function makeOffset(set_in, set_out)

    in_centroid = set_in[1]:clone():zero()
    out_centroid = in_centroid:clone():zero()
    for i = 1,(#set_in)[1] do
        in_centroid = in_centroid + set_in[i]/set_in[i]:norm()
        out_centroid = out_centroid + set_out[i]/set_out[i]:norm()
    end
    return out_centroid-in_centroid
end    

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')

cmd:option('-prefix','_','prefix for output')
cmd:option('-outputDir','./','where to put serialized params')
cmd:option('-trainPath','../pairFiles','where to find word pairs')
cmd:option('-devPath','../pairFiles','where to find word pairs')
cmd:option('-synsetPath','../../SynsetLists.th','where to find synsets')


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

train = readSet(cmdparams.inputSize, cmdparams.trainPath,cmdparams.synsetPath) 
train_in=train[1]
train_out=train[2]

--dev = readSet(cmdparams.inputSize, cmdparams.devPath) 
--dev_in=dev[1]
--dev_out=dev[2]

offset = makeOffset(train_in, train_out)

torch.save(output_path .. '.th' , offset)

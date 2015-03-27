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
            local vin = Vectors[win]
            local vout = Vectors[wout]
            if vin~= nil and vout~=nil then
                if set_in ==nil then
                    set_in = vin:clone()/vin:norm()
                    set_out= vout:clone()/vout:norm()
                else
                    set_in = torch.cat(set_in,vin/vin:norm(),2)
                    set_out = torch.cat(set_out,vout/vout:norm(),2)
                end
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
    return (out_centroid-in_centroid)/(#set_in)[1]
end    

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-outputPath','./','where to put serialized params')
cmd:option('-trainPath','../pairFiles','where to find word pairs')
cmd:option('-vectorPath','../pairFiles','where to find vector')

cmdparams = cmd:parse(arg)

-- Create dataset

Vectors = torch.load(cmdparams.vectorPath)

train = readSet(cmdparams.inputSize, cmdparams.trainPath) 
train_in=train[1]
train_out=train[2]

offset = makeOffset(train_in, train_out)

torch.save(cmdparams.outputPath .. '.th' , offset)

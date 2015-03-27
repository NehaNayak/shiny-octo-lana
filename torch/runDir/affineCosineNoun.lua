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
            print(wout)
            print(win)
            local vin = Vectors[win]
            local vout = Vectors[wout]
            if vin~= nil and vout~=nil then
                vin = vin:typeAs(m)
                vout = vout:typeAs(m)
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

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')

cmd:option('-outputPath','./','where to put serialized params')
cmd:option('-trainPath','../pairFiles','where to find word pairs')
cmd:option('-devPath','../pairFiles','where to find word pairs')

cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-iterLimit',10e3,'maximum number of iterations')

cmd:option('-useGlove',false,'whether to use Glove or word2vec')
cmd:option('-vectorPath','../pairFiles','where to find vector')

cmdparams = cmd:parse(arg)

if cmdparams.useGlove then
    vecset = 'g'
else
    vecset = 'v'
end

local output_path = table.concat({
				cmdparams.outputPath, 
				'_lr',
				cmdparams.learningRate,
				'_il',
				cmdparams.iterLimit,
				},"")

-- Create Train and Dev sets 

Vectors = torch.load(cmdparams.vectorPath)

train = readSet(cmdparams.inputSize, cmdparams.trainPath) 
train_in=train[1]
train_out=train[2]

dev = readSet(cmdparams.inputSize, cmdparams.devPath) 
dev_in=dev[1]
dev_out=dev[2]

-- Define model

model = nn.Sequential() -- define the container
model:add(nn.Linear(cmdparams.inputSize, cmdparams.inputSize)) -- define the only module
criterion = nn.CosineEmbeddingCriterion()

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_in)[1] then _nidx_ = 1 end

   local input = train_in[_nidx_]:clone()
   local target = train_out[_nidx_]:clone()

   dl_dx:zero()

   local loss_x = criterion:forward({model:forward(input), target},1)
   model:backward(input, criterion:backward({model.output, target},1)[1])

   return loss_x, dl_dx

end

sgd_params = {
   learningRate = cmdparams.learningRate,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

local prev_dev_loss=1000
local dev_loss=1000
bestLoss=1000
lessCount=0

ending=false
lessCount=0

for i = 1,cmdparams.iterLimit do
    train_loss = 0
    for j = 1,(#train_in)[1] do
        _,fs = optim.adagrad(feval,x,sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / (#train_in)[1]
    
    prev_dev_loss=dev_loss
    dev_loss=0
    for j = 1,(#dev_in)[1] do
        local input_sample = dev_in[j]
        local target_sample = dev_out[j]
        local loss = criterion:forward({model:forward(input_sample), target_sample},1)
        dev_loss = dev_loss+loss
    end
    dev_loss = dev_loss / (#dev_in)[1]
    
    --if dev_loss>prev_dev_loss then
    --    if dev_loss<bestLoss then
    --        bestLoss = dev_loss
    --        bestModel=model:clone()
    --    end
    --    lessCount=lessCount+1
    --    if lessCount==100 then
    --        torch.save(output_path .. '.th' , bestModel)
    --        break
    --    end
    --end

    if i%5==0 then
        print('current train loss = ' .. train_loss)
        print('current dev loss = ' .. dev_loss)
    end

end


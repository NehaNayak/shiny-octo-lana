f = assert(io.open("SynsetLists2.txt","r"))

curr = ""
list = {}
Synsets = {}


while true do
    line=f:read()
    if line==nil then
        break
    elseif line:sub(1,1)=="-" then
        Synsets[curr]=list
        curr = line:sub(2)
        list = {}
    else
        table.insert(list,line)
    end
end

torch.save("SynsetLists.th",Synsets)


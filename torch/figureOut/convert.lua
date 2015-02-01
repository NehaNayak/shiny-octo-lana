require('torch')

local path = arg[1]
local prefix_toks = stringx.split(path, '.')
table.remove(prefix_toks, #prefix_toks)
local prefix = stringx.join('.', prefix_toks)

print('prefix = ' .. prefix)

-- get dimension and number of lines
local file = io.open(path, 'r')
local line
local count = 0
local dim = 0
while true do
  line = file:read()
  if not line then break end
  if count == 0 then
    dim = #stringx.split(line) - 1
  end
  count = count + 1
end

print('count = ' .. count)
print('dim = ' .. dim)

-- convert to torch-friendly format
file:seek("set")
local vocab = io.open('glove.6B.vocab', 'w')
local vecs = torch.FloatTensor(count, dim)
for i = 1, count do
  if i % 10000 == 0 then print(i) end
  local tokens = stringx.split(file:read())
  local word = tokens[1]
  vocab:write(word .. '\n')
  for j = 1, dim do
    vecs[{i, j}] = tonumber(tokens[j + 1])
  end
end
file:close()
vocab:close()
torch.save(prefix .. '.th', vecs)

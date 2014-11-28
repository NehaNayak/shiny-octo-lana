import sys

vocab = set()

FILE_vocab = open(sys.argv[1],'r')
for line in FILE_vocab:
	vocab.add(line[:-1])

for line in sys.stdin:
	thisLine = line.split()
	if thisLine[0] in vocab:
		sys.stdout.write(line)

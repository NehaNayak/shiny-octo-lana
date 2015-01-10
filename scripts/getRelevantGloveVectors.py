import sys

requiredWords = set()

for line in open(sys.argv[1],'r'):
    requiredWords.add(line[:-1])

for line in sys.stdin:
    thisLine = line.split()
    if thisLine[0] in requiredWords:
        sys.stdout.write(line)

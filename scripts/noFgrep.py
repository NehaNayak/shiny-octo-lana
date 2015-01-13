import sys

Words = set()

for line in open(sys.argv[1],'r'):
    Words.add(line[:-1])

for line in sys.stdin:
    thisLine = line.split()
    if thisLine[0] in Words:
        sys.stdout.write(line)

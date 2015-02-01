import sys

gloveWords = set()

for line in sys.stdin:
    gloveWords.add(line[:-1])

for line in open(sys.argv[1],'r'):
    (hypo, hyper) = line.split()
    if hypo in gloveWords and hyper in gloveWords:
        sys.stdout.write(line)

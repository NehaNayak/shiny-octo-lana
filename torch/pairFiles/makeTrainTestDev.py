import sys
from random import shuffle

Pairs = []

for line in open(sys.argv[1],'r'):
    Pairs.append(line)

shuffle(Pairs)

Test=Pairs[:int(len(Pairs)*0.4)]
Dev=Pairs[int(len(Pairs)*0.4):int(len(Pairs)*0.5)]
Train=Pairs[int(len(Pairs)*0.5):]

with open(sys.argv[1].replace(".txt","_Train.txt"),'w') as f:
    for line in Train:
        f.write(line)

with open(sys.argv[1].replace(".txt","_Dev.txt"),'w') as f:
    for line in Dev:
        f.write(line)

with open(sys.argv[1].replace(".txt","_Test.txt"),'w') as f:
    for line in Test:
        f.write(line)



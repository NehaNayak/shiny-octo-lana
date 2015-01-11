import sys
import random

hypoFile = open(sys.argv[1],'r')
hyperFile = open(sys.argv[2],'r')
prefix = sys.argv[3]

hypo = []

for line in hypoFile:
    hypo.append(line)

hyper = []

for line in hyperFile:
    hyper.append(line)

listLen = len(hypo)

if not len(hyper)==listLen:
    print("FAIL")
    exit()

indices = range(listLen)
random.shuffle(indices)

hypoTest= [hypo[i] for i in indices[:int(0.4*listLen)]]
hypoDev = [hypo[i] for i in indices[int(0.4*listLen):int(0.5*listLen)]]
hypoTrain = [hypo[i] for i in indices[int(0.5*listLen):]]

hyperTest= [hyper[i] for i in indices[:int(0.4*listLen)]]
hyperDev = [hyper[i] for i in indices[int(0.4*listLen):int(0.5*listLen)]]
hyperTrain = [hyper[i] for i in indices[int(0.5*listLen):]]

print len(hypoTest),len(hypoDev), len(hypoTrain)
print len(hyperTest),len(hyperDev), len(hyperTrain)

with open(prefix+"_hypo_Test.txt",'w') as f:
    for line in hypoTest:
        f.write(line)
with open(prefix+"_hypo_Dev.txt",'w') as f:
    for line in hypoDev:
        f.write(line)
with open(prefix+"_hypo_Train.txt",'w') as f:
    for line in hypoTrain:
        f.write(line)
with open(prefix+"_hyper_Test.txt",'w') as f:
    for line in hyperTest:
        f.write(line)
with open(prefix+"_hyper_Dev.txt",'w') as f:
    for line in hyperDev:
        f.write(line)
with open(prefix+"_hyper_Train.txt",'w') as f:
    for line in hyperTrain:
        f.write(line)

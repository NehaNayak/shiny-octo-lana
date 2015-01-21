import sys

Actual = []
Other = []

for line in sys.stdin:
    (actual, other) =  line.split()
    Actual.append(int(actual))
    Other.append(int(other))

Correct = len(filter(lambda x:x ,map(lambda (x,y) : x>y,zip(a,b))))

print float(Correct)/len(Actual), float(sum(Actual))/len(Actual),float(sum(Other))/len(Other)

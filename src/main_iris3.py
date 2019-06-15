import sklearn

import matplotlib.pyplot as plt
from operator import itemgetter
import random

from src import inf

n = 100
m = 50000

arr = []
for i in range(1, n + 1):
    arr.append([])
    arr[-1].append(i)
    arr[-1].append(0)

arr2 = []
for i in range(m):
    X = []
    for j in range(n):
        X.append(random.randint(0, n))
    part_y = inf.part(X)
    arr[len(part_y)-1][1] += inf.entr(part_y)/m
    temp = sorted(arr, key=itemgetter(1), reverse=True)
    arr2.append([])
    arr2[-1].append(i)
    arr2[-1].append(temp[0][0])


    
plt.plot([x[0] for x in arr2], [x[1] for x in arr2])
plt.xlabel('s')
plt.ylabel('$k^*')
plt.show()

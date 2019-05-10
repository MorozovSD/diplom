#from sklearn import metrics
#from sklearn.metrics import pairwise_distances
#from sklearn import datasets
import numpy as np
#from sklearn.cluster import KMeans
import math
import random

def partitions(n):
    arr = [[]]
    temparr = []
    for i in range(n):
        arr[-1].append(1)
        temparr.append(1)
    while temparr[0] < n: 
        minx = min(temparr[:-1])
        indx = temparr.index(minx)
        temparr[-1] -= 1
        if temparr[-1] == 0:
            temparr.remove(0)
        temparr[indx] += 1
        sum = 0
        for i in range(indx + 1, len(temparr)):
            if temparr[i] > 1:
                temp = temparr.pop(i)
                for j in range(temp):
                    temparr.append(1)
                temparr.insert(i, 0)
        while 0 in temparr:
           temparr.remove(0)
        arr.append([])
        for i in range(len(temparr)):
            arr[-1].append(temparr[i])

        #lenarr = len(arr)
        #if lenarr%100000 == 0:
        #    print(lenarr)

    #print(lenarr)    
    return arr

def entropy(n, arr):
    entr = []
    for i in range(len(arr)):
        if len(arr[i]) == 1:
            entr.append([1])
        elif len(arr[i]) == n:
            entr.append([0])
        else:
            h = 0
            for j in range(len(arr[i])):
                h += arr[i][j]/n*math.log(arr[i][j], n)
            entr.append([h])
    return entr

def allocations(n, arr):
    alloc = []
    sum = 0
    total = n**n
    for i in range(len(arr)):
        alloc.append([])
        perm = 1
        for j in range(len(arr[i])):
            perm *= math.factorial(arr[i][j])
        perm = math.factorial(n)/perm
        comb = math.factorial(n)/(math.factorial(len(arr[i]))*math.factorial(n - len(arr[i]))) 
        rep = 1
        uniq = []
        numrep = []
        for j in range(len(arr[i])):
            if arr[i][j] not in uniq:
                uniq.append(arr[i][j])
        for j in range(len(uniq)):
            numrep.append(arr[i].count(uniq[j]))
        for j in range(len(numrep)):
            rep *= math.factorial(numrep[j])
        rep = math.factorial(len(arr[i]))/rep
        temp = perm*comb*rep
        sum += temp
        alloc[-1].append(temp)
    #if sum != total:
    #    print('error')
    #    print(total)
    #    print(sum)
    return alloc

def wentropy(n, part, entr, alloc):
    wentr = []
    for i in range(len(part)):
        wentr.append((alloc[i][0]/n**n)*entr[i][0])
    return wentr

def meanentropy(wentr):
    mean = 0
    for i in range(len(wentr)):
        mean += wentr[i]
    return mean

def getinformpart(mentr, part, entr):
    arr = []
    for i in range(len(entr)):
        if entr[i][0] > mentr:
            arr.append(part[i])
    return arr

def getbalancedinformpart(mentr, part, entr):
    entr.sort()
    i = 0
    templ = 0
    while templ < mentr:
        templ = entr[i][0]
        i += 1
    j = len(entr) - 1
    tempg = 1
    while tempg > mentr:
        tempg = entr[j][0]
        j -= 1
    if abs(mentr - tempg) >= abs(mentr - templ):
        optpart = i - 1
    else:
        optpart = j + 1
    return optpart

def getmaxwentr(wentr, part):
    maxwentr = max(wentr)
    ind = wentr.index(maxwentr)
    return ind
    
def getentr2(n, part):
    entr2 = 0
    alloc = allocations(n, part)
    for i in range(len(alloc)):
        if alloc[i][0] == 1:
            entr2 += 0
        else:
            entr2 += (alloc[i][0]/n**n)*math.log(alloc[i][0], n**n)
    return entr2

def evalnum(n):
    part = partitions(n)
    alloc = allocations(n, part)
    entr = entropy(n, part)
    entr2 = getentr2(n, part)
    wentr = wentropy(n, part, entr, alloc)
    mentr = meanentropy(wentr)
    #informparts = informpart(mentr, part, entr)
    balancedpart = part[getbalancedinformpart(mentr, part, entr)]
    maxwentr = part[getmaxwentr(wentr, part)]
    print('partitions entropy = ' + str(entr2))
    print('mean entropy = ' + str(mentr))
    print('optimal partition is ' + str(balancedpart))
    print(len(balancedpart))
    print('maximal weighted informative partition is ' + str(maxwentr))
    print(len(maxwentr))

def evalpart(arr):
    arr.sort(reverse = True)
    n = 0
    for i in range(len(arr)):
        n += arr[i]
    print(n)
    part = partitions(n)
    print(part)
    alloc = allocations(n, part)
    print(alloc)
    #for i in range(len(alloc)):
    #    print(alloc[i][0]/n**n)
    entr = entropy(n, part)
    print(entr)
    entr2 = getentr2(n, part)
    print(entr2)
    wentr = wentropy(n, part, entr, alloc)
    print('wentr')
    print(wentr)
    mentr = meanentropy(wentr)
    print(mentr)
    #informparts = informpart(mentr, part, entr)
    balancedpartentr = entr[getbalancedinformpart(mentr, part, entr)][0]
    print(balancedpartentr)
    maxwentr = getmaxwentr(wentr, part)
    print(maxwentr)
    
    ind = part.index(arr)
    print(ind)
    partentr = entr[ind][0]
    print('partition entropy = ' + str(partentr))
    print('difference from mean entropy = ' + str(partentr - mentr))
    print('difference from optimal partition entropy = ' + str(partentr - balancedpartentr))
    partalloc = alloc[ind][0]
    print('partition probability = ' + str(partalloc))
    partentr2 = math.log(partalloc, n**n)
    print('partition informativity = ' + str(partentr2))
    print('difference from partition entropy = ' + str(partentr2 - entr2))
    partwentr = (alloc[ind][0]/n**n)*partentr
    print('partition weighted entropy = ' + str(partwentr))
    print('mean weighted entropy = ' + str(mentr))
    print('maximal weighted entropy = ' + str(wentr[maxwentr]))
    print('difference from maximal weighted entropy = ' + str(partwentr - wentr[maxwentr]))
    print('difference from mean weighted entropy = ' + str(partentr - mentr))

 
"""dataset = datasets.load_iris()
X = dataset.data
rX = []
for i in range(10):
    r = random.randint(1,149)
    rX.append(list(X[r]))

for n in range(2, 11):
    kmeans_model = KMeans(n_clusters=n).fit(rX)
    labels = kmeans_model.labels_
    
    
    uniq = []
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i] not in uniq:
            uniq.append(labels[i])
    num = []
    for i in range(len(uniq)):
        num.append(labels.count(uniq[i]))
    inf = []
    total = len(labels)
    entr = 0
    for i in range(len(num)):
        entr += num[i]/total*math.log(num[i], total)

    
    total1 = total**total
    
    perm = 1
    for j in range(len(num)):
        perm *= math.factorial(num[j])
    perm = math.factorial(total)/perm

    comb = math.factorial(total)/(math.factorial(len(num))*math.factorial(total - len(num))) 

    rep = 1
    uniq = []
    numrep = []
    for j in range(len(num)):
        if num[j] not in uniq:
            uniq.append(num[j])
    for j in range(len(uniq)):
        numrep.append(num.count(uniq[j]))
    for j in range(len(numrep)):
        rep *= math.factorial(numrep[j])
    rep = math.factorial(len(num))/rep

    p=perm*comb*rep/total1

    print(n)
    print(entr)
    print(format(p, 'f'))
    print(format(p*entr, 'f'))
    print(evalpart(num))
    print(metrics.calinski_harabaz_score(rX, labels))

    
    for i in range(len(rX)):
        print(rX[i], end='')
        print(labels[i])"""

arr = [25, 25, 20]
evalpart(arr)
#evalnum(71)

        

    
        

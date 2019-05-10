#from sklearn import metrics
#from sklearn.metrics import pairwise_distances
#from sklearn import datasets
import numpy as np
#from sklearn.cluster import KMeans
import math
import random

def comb(n):
    arr = []
    for i in range(n):
        arr.append([])
        l = i + 1
        arr[-1].append(l)
        arr[-1].append(math.factorial(n)/(math.factorial(l)*math.factorial(n-l)))
    return arr

def H1(arr, n):
    tempnum = 0
    temph = 0
    i = 1
    m=arr[0]
    while tempnum<=n:
        if tempnum+m**i < n:
            tempnum+=m**i
            temph += math.log(i,n)*(m**i/n)
        else:
            temph += math.log(i,n)*((n-tempnum)/n)
            break
        i+=1
    arr.append(temph)

def P2(arr):
    temp = 0
    for i in range(len(arr)):
        temp += arr[i][1]
    for i in range(len(arr)):
        arr[i].append(arr[i][1]/temp)

def H2(arr):
    temph = 0
    for i in range(len(arr)):
        temph += arr[i][2]*arr[i][3]
        arr[i].append(arr[i][2]*arr[i][3])
    return temph

def main(n):
    arr = comb(n)
    for i in range(len(arr)):
        H1(arr[i], n)
    P2(arr)
    h2 = H2(arr)
    
    for i in range(len(arr)):
        print(arr[i])

    print(h2)    

main(256)

        

    
        

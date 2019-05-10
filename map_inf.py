import numpy as np
from sklearn.tree import DecisionTreeClassifier
from math import log

def map_inf(estimator, X, k):

    n = pow(k, len(X))
    
    dp = estimator.decision_path(X).toarray()

    inf = []
    for sample in dp:
        inf.append(log(pow(2, sum(sample) - 1)*k, n))

    return np.mean(inf)
        
        
        
    

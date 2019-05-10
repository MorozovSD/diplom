from math import log
import numpy as np 
from scipy.special import factorial

def fact(X):

    matrix = X.transpose()
    return list(map(lambda x: len(set(x)), matrix)) 

def entr(fact):

    n = sum(fact)

    inf = list(map(lambda x: log(x, n), fact))

    return np.mean(inf)
    
def fact_num(fact):

    return factorial(len(fact), exact = True)


def dom_inf(X):

    f = fact(X)
    f_entr = entr(f)
    #f_num = fact_num(f)
    
    #return Decimal(f_num)*Decimal(f_entr)

    return f_entr

from sklearn import datasets
from sklearn.cluster import KMeans
from multiprocessing import Process
import time
from operator import itemgetter
import matplotlib.pyplot as plt

from src import inf


def num_inf(n, k):
    k, y2_expentr = inf.num_inf_bayes(n, k)
    print(k, end=', ')
    print("{:.2E}".format(y2_expentr))
    
if __name__ == '__main__':ve

    iris = datasets.load_iris()
    X = iris.data
    y3 = iris.target
    # n = len(y3)
    n = 10

    model = KMeans(n_clusters=2).fit(X)
    y2 = model.predict(X).tolist()
    y2_entr, y2_prob, y2_probentr = inf.cod_inf(y2)
    print(inf.part(y2))
    print("{:.2f}".format(y2_entr))
    print("{:.2E}".format(y2_prob))
    print("{:.2E}".format(y2_probentr))

    y3_entr, y3_prob, y3_probentr = inf.cod_inf(y3)
    print(inf.part(y3))
    print("{:.2f}".format(y3_entr))
    print("{:.2E}".format(y3_prob))
    print("{:.2E}".format(y3_probentr))

    model = KMeans(n_clusters=4).fit(X)
    y4 = model.labels_.tolist()
    y4_entr, y4_prob, y4_probentr = inf.cod_inf(y4)
    print(inf.part(y4))
    print("{:.2f}".format(y4_entr))
    print("{:.2E}".format(y4_prob))
    print("{:.2E}".format(y4_probentr))

    p1 = Process(target=num_inf, args=(n, 2,))
    p1.start()

    p2 = Process(target=num_inf, args=(n, 3,))
    p2.start()

    p3 = Process(target=num_inf, args=(n, 4,))
    p3.start()

    p1.join()
    p2.join()
    p3.join()
    
    input()

import copy
import time
from collections import Counter
from scipy.special import factorial, comb
from math import log
import numpy as np
from decimal import Decimal
from multiprocessing import Process, Queue
from timeit import default_timer as timer

import cProfile

import itertools


# numba
# from graph_writer import to_graph


def parts(n):
    a = [1] * n
    y = -1
    v = n
    while v > 0:
        v -= 1
        x = a[v] + 1
        while y >= 2 * x:
            a[v] = x
            y -= x
            v += 1
        w = v + 1
        while x <= y:
            a[v] = x
            a[w] = y
            yield a[:w + 1]
            x += 1
            y -= 1
        a[v] = x + y
        y = a[v] - 1
        yield a[:w]

def gen_partitions_ms(n):
    """Generate all partitions of integer n (>= 0).

    Each partition is represented as a multiset, i.e. a dictionary
    mapping an integer to the number of copies of that integer in
    the partition.  For example, the partitions of 4 are {4: 1},
    {3: 1, 1: 1}, {2: 2}, {2: 1, 1: 2}, and {1: 4}.  In general,
    sum(k * v for k, v in a_partition.iteritems()) == n, and
    len(a_partition) is never larger than about sqrt(2*n).

    Note that the _same_ dictionary object is returned each time.
    This is for speed:  generating each partition goes quickly,
    taking constant time independent of n.
    """

    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        yield {}
        return

    ms = {n: 1}
    keys = [n]  # ms.keys(), from largest to smallest
    yield ms

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
        else:
            reuse = 0

        # Let i be the smallest key larger than 1.  Reuse one
        # instance of i.
        i = keys[-1]
        newcount = ms[i] = ms[i] - 1
        reuse += i
        if newcount == 0:
            del keys[-1], ms[i]

        # Break the remainder into pieces of size i-1.
        i -= 1
        q, r = divmod(reuse, i)
        ms[i] = q
        keys.append(i)
        if r:
            ms[r] = 1
            keys.append(r)

        yield ms

def partitions2(n, k=None):
    """Generate all partitions of integer n (>= 0) using integers no
    greater than k (default, None, allows the partition to contain n).

    Each partition is represented as a multiset, i.e. a dictionary
    mapping an integer to the number of copies of that integer in
    the partition.  For example, the partitions of 4 are {4: 1},
    {3: 1, 1: 1}, {2: 2}, {2: 1, 1: 2}, and {1: 4} corresponding to
    [4], [1, 3], [2, 2], [1, 1, 2] and [1, 1, 1, 1], respectively.
    In general, sum(k * v for k, v in a_partition.iteritems()) == n, and
    len(a_partition) is never larger than about sqrt(2*n).

    Note that the _same_ dictionary object is returned each time.
    This is for speed:  generating each partition goes quickly,
    taking constant time independent of n. If you want to build a list
    of returned values then use .copy() to get copies of the returned
    values:

    >>> p_all = []
    >>> for p in partitions(6, 2):
    ...         p_all.append(p.copy())
    ...
    >>> print p_all
    [{2: 3}, {1: 2, 2: 2}, {1: 4, 2: 1}, {1: 6}]

    Reference
    ---------
    Modified from Tim Peter's posting to accomodate a k value:
    http://code.activestate.com/recipes/218332/
    """

    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        yield {}
        return

    if k is None or k > n:
        k = n

    q, r = divmod(n, k)
    ms = {k : q}
    keys = [k]
    if r:
        ms[r] = 1
        keys.append(r)
    yield ms

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
        else:
            reuse = 0

        # Let i be the smallest key larger than 1.  Reuse one
        # instance of i.
        i = keys[-1]
        newcount = ms[i] = ms[i] - 1
        reuse += i
        if newcount == 0:
            del keys[-1], ms[i]

        # Break the remainder into pieces of size i-1.
        i -= 1
        q, r = divmod(reuse, i)
        ms[i] = q
        keys.append(i)
        if r:
            ms[r] = 1
            keys.append(r)

        yield ms

def test():
    start2 = time.time()
    x = gen_partitions_ms(150)
    # for i in x:
    #     print(i)
    length = sum(1 for el in x)
    # my_array = np.array()
    # for i, el in enumerate(x):
    #     my_array[i] = el
    # print(len(my_array))
    print(length)
    stop2 = time.time() - start2
    print(stop2)

# https://www.dcode.fr/partitions-generator
if __name__ == "__main__":
    # x = partitions(40)
    # time_serioes1 = {}
    # for i in range(1, 100, 5):
    #     start = time.time()
    #     length = sum(1 for el in gen_partitions_ms(i))
    #     print(length)
    #
    #     time_serioes1[i] = time.time() - start
    # to_graph(time_serioes1, title='Скорость работы', x_label='Число', y_label='Время разбиения')
    # print('--------------------------------')
    # print('--------------------------------')
    # print('--------------------------------')
    # print(time_serioes1)
    #
    #
    # time_serioes2 = {}
    # for i in range(60):
    #     start = time.time()
    #     print(len(list(partition_1(i))))
    #     time_serioes2[i] = time.time() - start
    #
    # to_graph(time_serioes2)

    # start1 = time.time()
    # print(len(list(partitions2(10))))
    # p_all = []

    # for p in partitions2(50, 3):
    #     p_all.append(p.copy())
    #
    # for item in sorted(p_all, key=lambda k: sum(k.keys())):
    #     print(f' {sum(item.keys())} - {item}')
    #
    # stop1 = time.time() - start1
    # print(stop1)

    # cProfile.run('test()')
    test()

    # start2 = time.time()
    # x = parts(75)
    # print(len(list(x)))
    # for i in x:
    #     print(i)
    # stop2 = time.time() - start2
    # print(stop2)

    # start3 = time.time()
    # print(len(list(parts(77))))
    # stop3 = time.time() - start3
    #
    # print(stop1, stop3)

    # for i in partitions(70):
    #     print(i)
    # print(time.time() - start)


# Преобрабазование формата
def part(X):
    M = np.array(X)
    shape = M.shape
    if len(shape) == 1:
        return list(Counter(X).values())
    else:
        m = shape[1]
        arr = []
        for i in range(m):
            arr.append(len(part(M[:, i])))
        return arr


# Подсчет вероятности
def perm_num(part):
    prod = 1
    for x in part:
        prod *= factorial(x)

    return factorial(sum(part)) / prod


def comb_num(part):
    return comb(sum(part), len(part))


def part_perm_num(part):
    part2 = Counter(part).values()

    return perm_num(part2)


def part_num(part):
    y_perm_num = perm_num(part)
    y_comb_num = comb_num(part)
    y_part_perm_num = part_perm_num(part)

    return y_perm_num * y_comb_num * y_part_perm_num


def part_prob(part):
    y_perm_num = perm_num(part)
    y_comb_num = comb_num(part)
    y_part_perm_num = part_perm_num(part)

    n = sum(part)

    return Decimal(Decimal(y_perm_num * y_comb_num * y_part_perm_num) / Decimal(n ** n))


# Подсчет энтропии
def entr(part):
    n = sum(part)

    inf = [log(x, n) for x in part]

    return np.mean(inf)


# Получение массива с количеством подмножеств и средней энтропией
def get_k_arr(n):
    arr = []
    for i in range(1, n + 1):
        arr.append([i, 0])

    _parts = parts(n)
    for _part in _parts:
        _entr = entr(_part)
        _part_prob = part_prob(_part)
        arr[len(_part) - 1][1] += _part_prob * Decimal(_entr)

    return arr

# Не используется
# def cod_inf(y):
#
#     y_part = part(y)
#     y_entr = entr(y_part)
#     y_part_prob = part_prob(y_part)
#
#     return y_entr, y_part_prob, Decimal(y_entr)*y_part_prob
#
# def num_inf_bayes(n, k):
#
#     _mean = 0
#     _parts = parts(n)
#     for _part in _parts:
#         if len(_part) == k:
#             _entr = entr(_part)
#             _part_prob = part_prob(_part)
#             _mean += _part_prob*Decimal(_entr)
#
#     print(k, end=', ')
#     print("{:.2E}".format(_mean))
#
#     return k, _mean
#
# def num_inf_laplas(n, k):
#
#     i = 0
#     _sum = 0
#     _parts = parts(n)
#     for _part in _parts:
#         i += 1
#         if len(_part) == k:
#             _sum += entr(_part)
#
#     _mean = _sum/i
#
#     print(k, end=', ')
#     print("{:.2E}".format(_mean))
#
#     return k, _mean
#
# def num_inf_wald(n, k):
#
#     arr = []
#     _min = 0
#     _parts = parts(n)
#     for _part in _parts:
#         if len(_part) == k:
#             _entr = entr(_part)
#             if _min == 0 or _entr < _min:
#                 _min = _entr
#
#     return _min
#
# def get_inf_arr(n, k, queue):
#
#     queue.put(num_inf_bayes(n, k))
#
# def get_k(n):
#
#     jobs = []
#     for i in range(1, n+1):
#         p = Process(target=num_inf_bayes, args=(n, i))
#         p.start()
#         jobs.append(p)
#     for p in jobs:
#         p.join()
#
# def get_k_arr_l(n):
#
#     arr = []
#     for i in range(1, n+1):
#         arr.append([i, 0])
#
#     _parts = parts(n)
#     i = 0
#     for _part in _parts:
#         i += 1
#         _entr = entr(_part)
#         arr[len(_part)-1][1] += _entr
#
#     for line in arr:
#         line[1] /= line[1]/i
#
#     return arr

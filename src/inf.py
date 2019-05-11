import copy
import logging
import sys
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

logger = logging.getLogger(__name__)
__author__ = 'Ildar Baimuratov (baimuratov.i@gmail.com)'


# def parts(n):
#     a = [1] * n
#     y = -1
#     v = n
#     while v > 0:
#         v -= 1
#         x = a[v] + 1
#         while y >= 2 * x:
#             a[v] = x
#             y -= x
#             v += 1
#         w = v + 1
#         while x <= y:
#             a[v] = x
#             a[w] = y
#             yield a[:w + 1]
#             x += 1
#             y -= 1
#         a[v] = x + y
#         y = a[v] - 1
#         yield a[:w]

def parts(n):
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
    part_max = {}
    part_min = {}
    _max = 0
    _min = sys.maxsize
    for i in range(1, n + 1):
        arr.append([i, 0])

    _parts = parts(n)
    for _part in _parts:
        part_len = len(_part) - 1

        _entr = entr(_part)
        _part_prob = part_prob(_part)

        prob_dec =_part_prob * Decimal(_entr)

        arr[part_len][1] += prob_dec

        if part_max.get(part_len):
            if prob_dec > part_max[part_len]:
                part_max[part_len] = prob_dec
                logger.info(f'Max entropy  for {part_len} changed from {part_max.get(part_len)} to {prob_dec}')
        else:
            part_max[part_len] = prob_dec

        if part_min.get(part_len):
            if prob_dec < part_min[part_len]:
                part_min[part_len] = prob_dec
                logger.info(f'Min entropy  for {part_len} changed from {part_min.get(part_len)} to {prob_dec}')
        else:
            part_min[part_len] = prob_dec

        if _min > prob_dec:
            _min = prob_dec
            logger.info(f'Min entropy changed from {part_min.get(part_len)} to {prob_dec}')

        if _max < prob_dec:
            _min = prob_dec
            logger.info(f'Max entropy changed from {part_max.get(part_len)} to {prob_dec}')

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

from collections import Counter
from scipy.special import factorial, comb
from math import log
import numpy as np


def part(y):
    return list(Counter(y).values())


def perm_num(part):
    prod = 1
    for x in part:
        prod *= factorial(x)

    return factorial(sum(part)) / prod


def comb_num(part):
    return comb(sum(part), len(part))


def part_perm_num(part):
    part2 = list(Counter(part).values())

    return perm_num(part2)


def part_num(part):
    y_perm_num = perm_num(part)
    y_comb_num = comb_num(part)
    y_part_perm_num = part_perm_num(part)

    return y_perm_num * y_comb_num * y_part_perm_num


def entr(part):
    n = sum(part)

    inf = list(map(lambda x: log(x, n), part))

    return np.mean(inf)


def cod_inf(y):
    y_part = part(y)
    y_entr = entr(y_part)
    return y_entr

    # y_part_num = part_num(y_part)
    # return y_entr*y_part_num

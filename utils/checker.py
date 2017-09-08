# -*- coding: utf-8 -*-

from __future__ import division
from sys import exit

import math
import os
import re


def is_nan_or_inf(n):
    return math.isnan(n) or math.isinf(n)


def is_number(x):
    try:
        f = float(x)
        return not is_nan_or_inf(f)
    except ValueError:
        return False
    except TypeError:
        return False


def is_date(w):
    if (not (bool(re.search("[a-z0-9]", w, re.IGNORECASE)))):
        return False
    if (len(w) != 10):
        return False
    if (w[4] != "-"):
        return False
    if (w[7] != "-"):
        return False
    for i in range(len(w)):
        if (not (w[i] == "X" or w[i] == "x" or w[i] == "-" or re.search("[0-9]", w[i]))):
            return False
    return True


def is_money(word):
    if (not (bool(re.search("[a-z0-9]", word, re.IGNORECASE)))):
        return False
    for i in range(len(word)):
        if (not (word[i] == "E" or word[i] == "." or re.search("[0-9]", word[i]))):
            return False
    return True


def is_number_column(a):
    for w in a:
        if len(w) != 1:
            return False
        if not is_number(w[0]):
            return False
    return True


def convert_table(table):
    a = []
    for i in xrange(len(table)):
        temp = []
        for j in xrange(len(table[i])):
            temp.append(''.join([str(w) for w in table[i][j]]))
        a.append(temp)
    return a

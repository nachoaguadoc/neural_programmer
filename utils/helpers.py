# -*- coding: utf-8 -*-

from sys import exit
from checker import *
from chars import *

import re
import os
import cPickle as pickle


def save_to(vars, path):
    with open(path, 'wb') as f:
        pickle.dump(vars, f)


def load_from(path):
    with open(path, 'r') as f:
        var = pickle.load(f)
    return var


def remove_conse(ner_tags, ner_values):
    for i in range(len(ner_tags)):
        if (ner_tags[i] == "NUMBER" or ner_tags[i] == "MONEY" or \
            ner_tags[i] == "PERCENT" or ner_tags[i] == "DATE") and \
            i + 1 < len(ner_tags) and ner_tags[i] == ner_tags[i + 1] and \
            ner_values[i] == ner_values[i + 1] and ner_values[i]!= "":
            word = ner_values[i]
            word = word.replace(">", "").replace("<", "").replace("=", "").replace(
                                "%", "").replace("~", "").replace("$", "").replace(
                                "£", "").replace("€", "")
            if (re.search("[A-Z]", word) and not (is_date(word)) and not (is_money(word))):
                ner_values[i] = "A"
            else:
                ner_values[i] = ","
    return ner_tags, ner_values


def prepro_sentence(tokens, ner_tags, ner_values, ann_wd_reject):
    sentence = []
    tokens = tokens.split('|')
    ner_tags = ner_tags.split('|')
    ner_values = ner_values.split('|')
    ner_tags, ner_values = remove_conse(ner_tags, ner_values)
    for i in xrange(len(tokens)):
        word = tokens[i]
        if ner_values[i]!='' and (ner_values[i] == 'NUMBER' or ner_values[i] == 'MONEY' or
                                  ner_values[i] == 'PERCENT' or ner_values[i] == 'DATE'):
            word = ner_values[i]
            word = word.replace(">", "").replace("<", "").replace("=", "").replace("%", "").replace("~", "").replace("$", "").replace("£", "").replace("€", "")
            if re.search('[A-Z]', word) and not is_date(word) and not is_money(word):
                word = tokens[i]
            if is_number(ner_values[i]):
                word = float(ner_values[i])
            elif is_number(word):
                word = float(word)
            if tokens[i] == 'score':
                word = 'score'
        if is_number(word):
            word = float(word)
        if not ann_wd_reject.has_key(word):
            if is_number(word) or is_date(word) or is_money(word):
                sentence.append(word)
            else:
                word = full_normalize(word)
                if not ann_wd_reject.has_key(word) and bool(re.search('[a-z0-9]', word, re.IGNORECASE)):
                    m = re.search(',', word)
                    sentence.append(word.replace(',', ''))
    if len(sentence) == 0:
        sentence.append('UNK')
    return sentence

# -*- coding: utf-8 -*-

from __future__ import division
from sys import exit

from utils.checker import *
from utils.chars import *
from utils.helpers import *

import math
import os
import re
import numpy as np
import tensorflow as tf
import cPickle as pickle

class WikiExample(object):
    """
    q_id:              question id
    q:                 questions
    a:                 answer
    tb_key:            table key
    lookup_mat:        lookup matrix
    is_bad_eg:         is bad example
    is_wd_lookup:      is word lookup
    is_ambi_wd_lookup: is ambiguous word lookup
    is_nb_lookup:      is number lookup
    is_nb_calc:        is number calc
    is_ukw_a:          is unknown answer
    """
    def __init__(self, id, q, a, tb_key):
        self.q_id = id
        self.q = q
        self.a = a
        self.tb_key = tb_key
        self.lookup_mat = []
        self.is_bad_eg = False
        self.is_wd_lookup = False
        self.is_ambi_wd_lookup = False
        self.is_nb_lookup = False
        self.is_nb_calc = False
        self.is_ukw_a = False


class TableInfo(object):
    """
    wd_cols:           word columns
    wd_col_names:      column names in parsed string
    wd_col_desc:       column descriptions
    wd_col_idx:        index to columns -> [0 ... #cols-1]
    nb_cols:           columns which have numbers in cell
    nb_col_names:      names of columns which have numbers in cell
    nb_col_desc:       descriptions of columns which have numbers in cell
    nb_col_idx:        index of columns which have numbers in cell
    p_wd_cols:         processed word columns
    p_nb_cols:         processed number columns
    o_cols:            orig columns in raw data
    """
    def __init__(self, wd_cols, wd_col_names, wd_col_desc, wd_col_idx, nb_cols, nb_col_names,
                 nb_col_desc, nb_col_idx, p_wd_cols, p_nb_cols, o_cols):
        self.wd_cols = wd_cols
        self.wd_col_names = wd_col_names
        self.wd_col_desc = wd_col_desc
        self.wd_col_idx = wd_col_idx
        self.nb_cols = nb_cols
        self.nb_col_names = nb_col_names
        self.nb_col_desc = nb_col_desc
        self.nb_col_idx = nb_col_idx
        self.p_wd_cols = p_wd_cols
        self.p_nb_cols = p_nb_cols
        self.o_cols = o_cols


class WikiQuestionLoader(object):
    def __init__(self, data_name, root_folder):
        self.root_folder = root_folder
        self.data_path = os.path.join(self.root_folder, 'data/data')
        self.data_name = data_name
        self.examples  = []

    def load_qa(self):
        data_source = os.path.join(self.data_path, self.data_name)
        f = tf.gfile.GFile(data_source, 'r')
        id_regex = re.compile('\(id ([^\)]*)\)')
        for line in f:
            id_match = id_regex.search(line)
            _id = id_match.group(1)
            self.examples.append(_id)
        f.close()

    def nb_q(self):
        return len(self.examples)

    def load(self):
        self.load_qa()


bad_nb = -999.0

class WikiQuestionGenerator(object):
    def __init__(self, train_name, dev_name, test_name, root_folder):
        self.train_name = train_name
        self.dev_name = dev_name
        self.test_name = test_name
        self.train_loader = WikiQuestionLoader(train_name, root_folder)
        self.dev_loader = WikiQuestionLoader(dev_name, root_folder)
        self.test_loader = WikiQuestionLoader(test_name, root_folder)
        self.bad_egs = 0
        self.root_folder = root_folder
        self.data_folder = os.path.join(self.root_folder, 'data/annotated/data')
        self.ann_egs = {}
        self.ann_tbs = {}
        self.custom_egs = {}
        self.custom_tbs = {}
        self.ann_wd_reject = {}
        self.ann_wd_reject['-lrb-'] = 1
        self.ann_wd_reject['-rrb-'] = 1
        self.ann_wd_reject['UNK'] = 1

    def prepro_sentence(self, tokens, ner_tags, ner_values):
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
            if not self.ann_wd_reject.has_key(word):
                if is_number(word) or is_date(word) or is_money(word):
                    sentence.append(word)
                else:
                    word = full_normalize(word)
                    if not self.ann_wd_reject.has_key(word) and bool(re.search('[a-z0-9]', word, re.IGNORECASE)):
                        m = re.search(',', word)
                        sentence.append(word.replace(',', ''))
        if len(sentence) == 0:
            sentence.append('UNK')
        return sentence

    def load_ann_data(self, in_file):
        self.ann_egs = {}
        self.ann_tbs = {}
        f =  tf.gfile.GFile(in_file, 'r')
        counter = 0
        for line in f:
            if counter > 0:
                line = line.strip()
                q_id, utt, context, tar_val, tokens,_, _, ner_tags, ner_values, tar_canon = line.split('\t')
                q = self.prepro_sentence(tokens, ner_tags, ner_values)
                tar_canon = tar_canon.split('|')
                self.ann_egs[q_id] = WikiExample(q_id, q, tar_canon, context)
                self.ann_tbs[context] = []
            counter += 1
        print 'Annotated egs loaded', len(self.ann_egs)
        f.close()

    def load_custom_tbs(self):
        self.custom_tbs = {}
        self.custom_tbs['csv/custom-csv/uefa.csv'] = []
        self.custom_tbs['csv/custom-csv/swisscom.csv'] = []
        print 'Custom tables loaded'

    def load_custom_data(self, q_id, input, context):
        input = input.replace("?", " ?")
        input = input.lower()    
        tokens = input.split(' ')
        ner_tags = ''
        ner_values = ''
        new_tokens = ''
        for t in tokens:
            if t.isdigit():
                ner_tags += 'NUMBER|'
                ner_values += str(float(t)) + '|'
            else:
                ner_tags += 'O|'
                ner_values += '|'
            new_tokens += t.encode('utf8') + '|'
        ner_tags = ner_tags[:-1]
        ner_values = ner_values[:-1]
        new_tokens = new_tokens[:-1]
        tar_canon = "UNK"
        q = self.prepro_sentence(new_tokens, ner_tags, ner_values)
        self.custom_egs[question_id] = WikiExample(q_id, q, tar_canon, context)
        return q


    def load_ann_tbs(self, custom=False, desc=False):
        """
        annotated tables keys are contex
        """
        tables = self.custom_tbs if mode=='custom' else self.ann_tbs
        for table in tables.keys():
            ann_tb = table.replace('csv', 'annotated')
            o_cols = []
            p_cols = []
            # print os.path.join(self.root_folder+'data', ann_tb)
            f = tf.gfile.GFile(os.path.join(self.root_folder+'data', ann_tb), 'r')
            counter = 0
            for line in f:
                if counter > 0:
                    line = line.strip()
                    line = line + '\t' * (13 - len(line.split('\t')))
                    row, col, _, _, tokens, _, _, \
                    ner_tags, ner_values, number, _, _, _ = line.split('\t')
                counter += 1
            f.close()
            max_row = int(row)
            max_col = int(col)
            for i in xrange(max_col + 1):
                o_cols.append([])
                p_cols.append([])
                for j in xrange(max_row + 1):
                    o_cols[i].append(bad_nb)
                    p_cols[i].append(bad_nb)
            f = tf.gfile.GFile(os.path.join(self.root_folder+'data', ann_tb), 'r')
            counter = 0
            col_names = []
            for line in f:
                if counter > 0:
                    line = line.strip()
                    line = line + '\t' * (13 - len(line.split('\t')))
                    row, col, _, content, tokens, _, _,\
                    ner_tags, ner_values, number, _, _, _ = line.split('\t')
                    if custom:
                        tokens = '|'.join(nltk.word_tokenize(content))
                        ner_tags = 'O|' * len(tokens)
                        ner_tags = ner_tags[:-1]
                        ner_values = '|' * (len(tokens)-1)
                    entry = self.prepro_sentence(tokens, ner_tags, ner_values)
                    if row=='-2':
                        new_entry = []
                        if desc:
                            pos_tags = nltk.pos_tag(entry)
                            for tag in pos_tags:
                              if tag[0] not in ['UNK', 'is', 'are']  and tag[1] in ['NN', 'JJ', 'NNS', 'NNP', 'NNPS','RB', 'SYM', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                                new_entry.append(tag[0])
                        print("Processing column descriptions:")
                        print("       Old entry: " + entry)
                        print("       New entry: " + new_entry)
                        col_desc.append(new_entry)
                    if row == '-1':
                        col_names.append(entry)
                    else:
                        o_cols[int(col)][int(row)] = entry
                        if len(entry) == 1 and is_number(entry[0]):
                            p_cols[int(col)][int(row)] = float(entry[0])
                        else:
                            for single_entry in entry:
                                if is_number(single_entry):
                                    p_cols[int(col)][int(row)] = float(single_entry)
                                    break
                            nt = ner_tags.split('|')
                            nv = ner_values.split('|')
                            for i_entry in xrange(len(tokens.split('|'))):
                                if nt[i_entry] == 'DATE' and is_number(nv[i_entry].replace("-", "").replace("X", "")):
                                    p_cols[int(col)][int(row)] = float(nv[i_entry].replace("-", "").replace("X", ""))
                        if len(entry) == 1 and (is_number(entry[0]) or is_date(entry[0]) or is_money(entry[0])):
                            if (len(entry) == 1 and not (is_number(entry[0])) and is_date(entry[0])):
                                entry[0] = entry[0].replace("X", "x")
                counter += 1
            wd_cols, p_wd_cols, wd_col_names, wd_col_idx, wd_col_desc = [], [], [], [], []
            nb_cols, p_nb_cols, nb_col_names, nb_col_idx, nb_col_desc = [], [], [], [], []
            for i in xrange(max_col + 1):
                if is_number_column(o_cols[i]):
                    nb_col_idx.append(i)
                    nb_col_names.append(col_names[i])
                    nb_col_desc.append(col_desc[i])
                    temp = []
                    for w in o_cols[i]:
                        if is_number(w[0]):
                            temp.append(w[0])
                    nb_cols.append(temp)
                    p_nb_cols.append(p_cols[i])
                else:
                    wd_col_idx.append(i)
                    wd_col_names.append(col_names[i])
                    wd_col_desc.append(col_desc[i])
                    wd_cols.append(o_cols[i])
                    p_wd_cols.append(p_cols[i])

            table_info = TableInfo(wd_cols, wd_col_names, wd_col_desc, wd_col_idx, nb_cols,
                                   nb_col_names, nb_col_desc, nb_col_idx, p_wd_cols, p_nb_cols, o_cols)
            tables[table] = table_info
            f.close()

    def answer_classification(self):
        lookup_questions = 0
        number_lookup_questions = 0
        word_lookup_questions = 0
        ambiguous_lookup_questions = 0
        number_questions = 0
        bad_questions = 0
        ice_bad_questions = 0
        tot = 0
        got = 0
        ice = {}
        with tf.gfile.GFile(self.root_folder + "data/arvind-with-norms-2.tsv", mode="r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if (not (self.ann_egs.has_key(line.split("\t")[0]))):
                    continue
                if (len(line.split("\t")) == 4):
                    line = line + "\t" * (5 - len(line.split("\t")))
                    if (not (is_number(line.split("\t")[2]))):
                        ice_bad_questions += 1
                (example_id, ans_index, ans_raw, process_answer, matched_cells) = line.split("\t")
                if (ice.has_key(example_id)):
                    ice[example_id].append(line.split("\t"))
                else:
                    ice[example_id] = [line.split("\t")]
        for q_id in self.ann_egs.keys():
            tot += 1
            example = self.ann_egs[q_id]
            table_info = self.ann_tbs[example.tb_key]
            # Figure out if the answer is numerical or lookup
            n_cols = len(table_info.o_cols)
            n_rows = len(table_info.o_cols[0])
            example.lookup_mat = np.zeros((n_rows, n_cols))
            exact_matches = {}
            for (example_id, ans_index, ans_raw, process_answer, matched_cells) in ice[q_id]:
                for match_cell in matched_cells.split("|"):
                    if (len(match_cell.split(",")) == 2):
                        (row, col) = match_cell.split(",")
                        row = int(row)
                        col = int(col)
                        if (row >= 0):
                            exact_matches[ans_index] = 1
            answer_is_in_table = len(exact_matches) == len(example.a)
            if (answer_is_in_table):
                for (example_id, ans_index, ans_raw, process_answer, matched_cells) in ice[q_id]:
                    for match_cell in matched_cells.split("|"):
                        if (len(match_cell.split(",")) == 2):
                            (row, col) = match_cell.split(",")
                            row = int(row)
                            col = int(col)
                            example.lookup_mat[row, col] = float(ans_index) + 1.0
            example.lookup_number_answer = 0.0
            if (answer_is_in_table):
                lookup_questions += 1
                if len(example.a) == 1 and is_number(example.a[0]):
                    example.number_answer = float(example.a[0])
                    number_lookup_questions += 1
                    example.is_nb_lookup = True
                else:
                    #print "word lookup"
                    example.calc_answer = example.number_answer = 0.0
                    word_lookup_questions += 1
                    example.is_wd_lookup = True
            else:
                if (len(example.a) == 1 and is_number(example.a[0])):
                    example.number_answer = example.a[0]
                    example.is_nb_calc = True
                else:
                    bad_questions += 1
                    example.is_bad_eg = True
                    example.is_ukw_a = True
            example.is_lookup = example.is_wd_lookup or example.is_nb_lookup
            if not example.is_wd_lookup and not example.is_bad_eg:
                number_questions += 1
                example.calc_answer = example.a[0]
                example.lookup_number_answer = example.calc_answer
            # Split up the lookup matrix into word part and number part
            nb_col_idx = table_info.nb_col_idx
            wd_col_idx = table_info.wd_col_idx
            example.wd_cols = table_info.wd_cols
            example.nb_cols = table_info.nb_cols
            example.wd_col_names = table_info.wd_col_names
            example.p_nb_cols = table_info.p_nb_cols
            example.p_wd_cols = table_info.p_wd_cols
            example.nb_col_names = table_info.nb_col_names
            example.number_lookup_mat = example.lookup_mat[:, nb_col_idx]
            example.word_lookup_mat = example.lookup_mat[:, wd_col_idx]
    
    def custom_answer_classification(self, q_id):

    eg = self.custom_egs[q_id]
    tb_info = self.custom_tbs[eg.tb_key]
    # Figure out if the answer is numerical or lookup
    n_cols = len(tb_info.o_cols)
    n_rows = len(tb_info.o_cols[0])
    eg.lookup_mat = np.zeros((n_rows, n_cols))

    # Split up the lookup matrix into word part and number part
    nb_col_idx = tb_info.nb_col_idx
    wd_col_idx = tb_info.wd_col_idx
    eg.wd_cols = tb_info.wd_cols
    eg.nb_cols = tb_info.nb_cols

    eg.wd_col_idx = tb_info.wd_col_idx
    eg.wd_col_names = tb_info.wd_col_names
    eg.wd_col_desc = tb_info.wd_col_desc
    eg.p_wd_cols = tb_info.p_wd_cols

    eg.nb_col_idx = tb_info.nb_col_idx
    eg.nb_col_names = tb_info.nb_col_names
    eg.nb_col_desc = tb_info.nb_col_desc
    eg.p_nb_cols = tb_info.p_nb_cols

    eg.number_lookup_mat = eg.lookup_mat[:, nb_col_idx]
    eg.word_lookup_mat = eg.lookup_mat[:, wd_col_idx]
    return eg

    def load_example(self, q_id, tokens, context):
        q = self.load_custom_data(q_id, tokens, context)
        example = self.custom_answer_classification(question_id)
        return example

    def load(self, mode=None, desc=False):
        train_data = []
        dev_data = []
        test_data = []
        self.load_ann_data(os.path.join(self.data_folder, "training.annotated"))
        self.load_ann_tbs(False, False)

        if (mode=='demo-console' or mode=='demo-visual' or mode=='custom-test'):
             self.load_custom_tbs()
             self.load_ann_tbs(True, desc)

        self.answer_classification()
        self.train_loader.load()
        self.dev_loader.load()

        for i in range(self.train_loader.nb_q()):
            example = self.train_loader.examples[i]
            example = self.ann_egs[example]
            train_data.append(example)
        for i in range(self.dev_loader.nb_q()):
            example = self.dev_loader.examples[i]
            dev_data.append(self.ann_egs[example])

        self.load_ann_data(os.path.join(self.data_folder, "pristine-unseen-tables.annotated"))
        self.load_ann_tbs()
        self.answer_classification()
        self.test_loader.load()
        for i in range(self.test_loader.nb_q()):
            example = self.test_loader.examples[i]
            test_data.append(self.ann_egs[example])

        return train_data, dev_data, test_data

if __name__ == '__main__':
    obj = WikiQuestionGenerator('random-split-1-train.examples', \
                                'random-split-1-dev.examples', \
                                'pristine-unseen-tables.examples', './')
    train, dev, test = obj.load()
    save_to(train, './p_data'+'/_train.pkl')
    save_to(dev, './p_data'+'/_dev.pkl')
    save_to(test, './p_data'+'/_test.pkl')
    print train[0].wd_cols

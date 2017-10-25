# -*- coding: utf-8 -*-
from __future__ import division
from sys import exit
import copy
from utils.checker import *
from utils.chars import *
from utils.helpers import *
import nltk
import math
import os
import re
import numpy as np
import tensorflow as tf
import warnings
class WikiExample(object):
    """
    q_id:              question id
    q:                 questions
    a:                 answer
    tb_key:            table key
    lookup_mat:        lookup matrix
    sim_tokens:        list of tokens that will be used for similarity search
    is_bad_eg:         is bad example
    is_wd_lookup:      is word lookup
    is_ambi_wd_lookup: is ambiguous word lookup
    is_nb_lookup:      is number lookup
    is_nb_calc:        is number calc
    is_ukw_a:          is unknown answer
    """
    def __init__(self, id, q, a, tb_key, q_og, sim_tokens):
        self.q_id = id
        self.q = q
        self.q_og = q_og
        self.a = a
        self.tb_key = tb_key
        self.sim_tokens = sim_tokens
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
        self.data_path = os.path.join(self.root_folder, 'data')
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
bad_count = 0

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
        self.data_folder = os.path.join(self.root_folder, 'annotated/data')
        self.ann_egs = {}
        self.ann_tbs = {}
        self.final_ann_tbs = {}
        self.custom_egs = {}
        self.custom_tbs = {}
        self.ann_wd_reject = {}
        self.ann_wd_reject['-lrb-'] = 1
        self.ann_wd_reject['-rrb-'] = 1
        self.ann_wd_reject['UNK'] = 1
        self.TOKENIZER_REGEX = re.compile(r'(?:(?:\s+|^)[^\w\d\$\&]*\s*|\s*[^\w\d\$\&]*(?:\s+|$)[^\w\d\$\&]*)')
    def prepro_sentence(self, tokens, ner_tags, ner_values):
        sentence = []
        tokens = tokens.split('|')
        ner_tags = ner_tags.split('|')
        ner_values = ner_values.split('|')
        ner_tags, ner_values = remove_conse(ner_tags, ner_values)
        for t in tokens:
            t = t.replace(">", "").replace("<", "").replace("=", "").replace("%", "").replace("~", "").replace("$", "").replace("£", "").replace("€", "")
            if t.isdigit():
                word = float(t)
            else:
                word = t
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


    def tokenize(self, text):
        warnings.simplefilter("ignore")
        return re.split(self.TOKENIZER_REGEX, text)
    def load_ann_data(self, in_file):
        self.ann_egs = {}
        self.ann_tbs = {}
        f =  tf.gfile.GFile(in_file, 'r')
        counter = 0
        for line in f:
            if counter > 0:
                line = line.strip()
                line_len = len(line.split('\t'))
                #if line_len == 10:
                #    q_id, utt, context, tar_val, tokens,_, pos_string, ner_tags, ner_values, tar_canon = line.split('\t')
                #    utt_de = utt
                #    tar_val_de = tar_canon
                #elif line_len == 11:
                #    q_id, utt, context, tar_val, tokens,_, pos_string, ner_tags, ner_values, tar_canon, utt_de = line.split('\t')
                #    tar_val_de = tar_canon
                if line_len < 12:
                    continue
                else:
                    q_id, utt, context, tar_val, tokens,_, pos_string, ner_tags, ner_values, tar_canon, utt_de, tar_val_de = line.split('\t')
                tokens_de = '|'.join(self.tokenize(utt_de))
                q_de = self.prepro_sentence(tokens_de, ner_tags, ner_values)
                print(q_de)
                sim_tokens = []
                pos_tags = pos_string.split('|')
                for i in range(len(pos_tags)):
                    tag = pos_tags[i]
                    if tag in ["NN", "NNS", "NNP", "JJ"]:
                        sim_tokens.append(tokens[i])
                tar_canon_de = []
                target_de = tar_val_de.split("|")
                for t in target_de:
                    target_de_word = t.split(" ")
                    tar_canon_word = []
                    for s in target_de_word:
                        if is_number(s):
                            tar_canon_word.append(str(float(s)))
                        else:
                            tar_canon_word.append(s)
                    tar_canon_de.append(' '.join(tar_canon_word))
                self.ann_egs[q_id] = WikiExample(q_id, q_de, tar_canon_de, context, utt_de, sim_tokens)
                self.ann_tbs[context] = []
            counter += 1
        print 'Annotated egs loaded', len(self.ann_egs)
        f.close()

    def load_custom_tbs(self):
        self.custom_tbs = {}
        self.custom_tbs['csv/custom-csv/swisscom.csv.translated'] = []

        print 'Custom tables loaded'

    def load_custom_data(self, q_id, input_og, context):
        input = input_og.replace("?", " ?")
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
        self.custom_egs[q_id] = WikiExample(q_id, q, tar_canon, context, input_og, [])
        return q 


    def load_ann_tbs(self, custom=False, desc=False):
        """
        annotated tables keys are contex
        """
        tables = self.custom_tbs if custom else self.ann_tbs
        for table in tables.keys():
            ann_tb = table.replace('csv', 'annotated')+ '.translated'
            o_cols = []
            p_cols = []
            # print os.path.join(self.root_folder+'data', ann_tb)
            f = tf.gfile.GFile(os.path.join(self.root_folder, ann_tb), 'r')
            counter = 0
            for line in f:
                if counter > 0:
                    line = line.strip()
                    line = line + '\t' * (14 - len(line.split('\t')))
                    row, col, _, _, tokens, _, _,  ner_tags, ner_values, number, _, _, _, content_de = line.split('\t')
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
            f = tf.gfile.GFile(os.path.join(self.root_folder, ann_tb), 'r')
            counter = 0
            col_names = []
            col_desc = []
            for line in f:
                if counter > 0:
                    line = line.strip()
                    line = line + '\t' * (14 - len(line.split('\t')))
                    row, col, _, content, tokens, _, _,\
                    ner_tags, ner_values, number, _, _, _, content_de = line.split('\t')
                    if content_de != '':
                        tokens = '|'.join(self.tokenize(content_de))
                    ner_tags = 'O|' * len(tokens)
                    ner_tags = ner_tags[:-1]
                    ner_values = '|' * (len(tokens)-1)
                    if not custom:
                        col_desc.append([])
                    entry = self.prepro_sentence(tokens, ner_tags, ner_values)
                    if row=='-2':
                        new_entry = []
                        if desc:
                        print("Processing column descriptions:")
                        print("       Old entry: " + str(entry))
                        print("       New entry: " + str(new_entry))
                            col_desc.append([str(e).lower() for e in entry])
                    if row == '-1':
                        col_names.append([str(e).lower() for e in entry])
                    else:
                        o_cols[int(col)][int(row)] = entry
                        if len(entry) == 1 and is_number(entry[0]):
                            p_cols[int(col)][int(row)] = float(entry[0])
                        else:
                            for single_entry in entry:
                                if is_number(single_entry):
                                    p_cols[int(col)][int(row)] = float(single_entry)
                                    break
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
#        with tf.gfile.GFile(self.root_folder + "arvind-with-norms-2-syns.tsv", mode="r") as f:
        with tf.gfile.GFile(self.root_folder + "arvind-with-norms-2.tsv", mode="r") as f:
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
            example.nb_col_idx = table_info.nb_col_idx
            example.wd_col_idx = table_info.wd_col_idx
            example.wd_cols = table_info.wd_cols
            example.wd_col_names = table_info.wd_col_names
            example.wd_col_desc = table_info.wd_col_desc
            example.p_wd_cols = table_info.p_wd_cols
            example.word_lookup_mat = example.lookup_mat[:, example.wd_col_idx]

            example.nb_cols = table_info.nb_cols
            example.nb_col_names = table_info.nb_col_names
            example.nb_col_desc = table_info.nb_col_desc
            example.p_nb_cols = table_info.p_nb_cols
            example.number_lookup_mat = example.lookup_mat[:, example.nb_col_idx]

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
        self.load_custom_data(q_id, tokens, context)
        example = self.custom_answer_classification(q_id)
        return example

    def load(self, mode=None, desc=False):
        train_data = []
        dev_data = []
        test_data = []
#        self.load_ann_data(os.path.join(self.data_folder, "training-syns.annotated"))
        self.load_ann_data(os.path.join(self.data_folder, "training.annotated.translated"))
        self.load_ann_tbs(False, False)
        if (mode=='demo-console' or mode=='demo-visual' or mode=='custom-test'):
             self.load_custom_tbs()
             self.load_ann_tbs(True, desc)
        self.final_ann_tbs = copy.deepcopy(self.ann_tbs) 
        self.answer_classification()
        self.train_loader.load()
        self.dev_loader.load()

        for i in range(self.train_loader.nb_q()):
            example = self.train_loader.examples[i]
            if example in self.ann_egs.keys():
                example = self.ann_egs[example]
                train_data.append(example)
        for i in range(self.dev_loader.nb_q()):
            example = self.dev_loader.examples[i]
            if example in self.ann_egs.keys():
                dev_data.append(self.ann_egs[example])

        #self.load_ann_data(os.path.join(self.data_folder, "pristine-unseen-tables.annotated"))
        #self.load_ann_tbs()
        #self.answer_classification()
        #self.test_loader.load()
        #for i in range(self.test_loader.nb_q()):
        #    example = self.test_loader.examples[i]
        #    test_data.append(self.ann_egs[example])

        return train_data, dev_data, test_data
if __name__ == '__main__':
    pass

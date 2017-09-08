# -*- coding: utf-8 -*-

from __future__ import division
from para import FLAGS
from random import Random
from utils.helpers import *
from sys import exit
from wikiqa import WikiExample, WikiQuestionLoader, TableInfo, WikiQuestionGenerator


import tensorflow as tf
import copy
import numbers
import numpy as np
import wikiqa


def return_index(a):
    for idx, ele in enumerate(a):
        if ele == 1.0:
            return idx


class Utility:
    def __init__(self):
        global FLAGS
        self.FLAGS = FLAGS
        self.unk_token = "UNK"
        self.entry_match_token = "entry_match"
        self.column_match_token = "column_match"
        self.dummy_token = "dummy_token"
        self.tf_data_type = {}
        self.tf_data_type["double"] = tf.float64
        self.tf_data_type["float"] = tf.float32
        self.np_data_type = {}
        self.np_data_type["double"] = np.float64
        self.np_data_type["float"] = np.float32
        self.operations_set = ["count", "prev", "next", "first_rs", "last_rs",
                               "group_by_max", "greater", "lesser", "geq", "leq",
                               "max", "min", "word-match", "reset_select", "print"]
        self.word_ids = {}
        self.reverse_word_ids = {}
        self.word_count = {}
        self.random = Random(FLAGS.python_seed)


def construct_vocab(data, utility, add_word=False):
    ans = []
    for example in data:
        sent = ""
        for word in example.q:
            if not isinstance(word, numbers.Number):
                sent += word + " "
        example.original_nc = copy.deepcopy(example.nb_cols)
        example.original_wc = copy.deepcopy(example.wd_cols)
        example.original_nc_names = copy.deepcopy(example.nb_col_names)
        example.original_wc_names = copy.deepcopy(example.wd_col_names)
        if add_word:
            continue
        number_found = 0
        if not example.is_bad_eg:
            for word in example.q:
                if isinstance(word, numbers.Number):
                    number_found += 1
                else:
                    if not utility.word_ids.has_key(word):
                        utility.words.append(word)
                        utility.word_count[word] = 1
                        utility.word_ids[word] = len(utility.word_ids)
                        utility.reverse_word_ids[utility.word_ids[word]] = word
                    else:
                        utility.word_count[word] += 1
            for col_name in example.wd_col_names:
                for word in col_name:
                    if isinstance(word, numbers.Number):
                        number_found += 1
                    else:
                        if not utility.word_ids.has_key(word):
                            utility.words.append(word)
                            utility.word_count[word] = 1
                            utility.word_ids[word] = len(utility.word_ids)
                            utility.reverse_word_ids[utility.word_ids[word]] = word
                        else:
                            utility.word_count[word] += 1
            for col_name in example.nb_col_names:
                for word in col_name:
                    if isinstance(word, numbers.Number):
                        number_found += 1
                    else:
                        if not utility.word_ids.has_key(word):
                            utility.words.append(word)
                            utility.word_count[word] = 1
                            utility.word_ids[word] = len(utility.word_ids)
                            utility.reverse_word_ids[utility.word_ids[word]] = word
                        else:
                            utility.word_count[word] += 1

def word_lookup(word, utility):
  if utility.word_ids.has_key(word):
    return word
  else:
    return utility.unk_token


def convert_to_int_2d_and_pad(a, utility):
    """
    trim input based on max length permitted.
    translate token to index according to dictionary
    """
    ans = []
    for b in a:
        temp = []
        if len(b) > utility.FLAGS.max_entry_length:
            b = b[0:utility.FLAGS.max_entry_length]
        for remaining in range(len(b), utility.FLAGS.max_entry_length):
            b.append(utility.dummy_token)
        assert len(b) == utility.FLAGS.max_entry_length
        for word in b:
            temp.append(utility.word_ids[word_lookup(word, utility)])
        ans.append(temp)
    return ans


def convert_to_bool_and_pad(a, utility):
    """
    get bool list for future masking.
    """
    a = a.tolist()
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (a[i][j] < 1):
                a[i][j] = False
            else:
                a[i][j] = True
        a[i] = a[i] + [False] * (utility.FLAGS.max_elements - len(a[i]))
    return a

def add_special_words(utility):
    utility.words.append(utility.entry_match_token)
    utility.word_ids[utility.entry_match_token] = len(utility.word_ids)
    utility.reverse_word_ids[utility.word_ids[utility.entry_match_token]] = utility.entry_match_token
    utility.entry_match_token_id = utility.word_ids[utility.entry_match_token]
    print "entry match token: ", utility.word_ids[utility.entry_match_token], utility.entry_match_token_id
    utility.words.append(utility.column_match_token)
    utility.word_ids[utility.column_match_token] = len(utility.word_ids)
    utility.reverse_word_ids[utility.word_ids[utility.column_match_token]] = utility.column_match_token
    utility.column_match_token_id = utility.word_ids[utility.column_match_token]
    print "entry match token: ", utility.word_ids[utility.column_match_token], utility.column_match_token_id
    utility.words.append(utility.dummy_token)
    utility.word_ids[utility.dummy_token] = len(utility.word_ids)
    utility.reverse_word_ids[utility.word_ids[utility.dummy_token]] = utility.dummy_token
    utility.dummy_token_id = utility.word_ids[utility.dummy_token]
    utility.words.append(utility.unk_token)
    utility.word_ids[utility.unk_token] = len(utility.word_ids)
    utility.reverse_word_ids[utility.word_ids[utility.unk_token]] = utility.unk_token


def perform_word_cutoff(utility):
    if (utility.FLAGS.word_cutoff > 0):
        for word in utility.word_ids.keys():
            if (utility.word_count.has_key(word) and utility.word_count[word] < \
                    utility.FLAGS.word_cutoff and word != utility.unk_token and \
                    word != utility.dummy_token and word != utility.entry_match_token and \
                    word != utility.column_match_token):
                utility.word_ids.pop(word)
                utility.words.remove(word)


def word_dropout(question, utility):
    if (utility.FLAGS.word_dropout_prob > 0.0):
        new_question = []
        for i in range(len(question)):
            if (question[i] != utility.dummy_token_id and utility.random.random() > utility.FLAGS.word_dropout_prob):
                new_question.append(utility.word_ids[utility.unk_token])
            else:
                new_question.append(question[i])
        return new_question
    else:
        return question


def generate_feed_dict(data, curr, batch_size, gr, train=False, utility=None):
    feed_dict = {}
    feed_examples = []

    for j in range(batch_size):
        feed_examples.append(data[curr + j])
    if train:
        feed_dict[gr.batch_question] = [word_dropout(feed_examples[j].question, utility) for j in range(batch_size)]
    else:
        feed_dict[gr.batch_question] = [feed_examples[j].question for j in range(batch_size)]

    feed_dict[gr.batch_question_attention_mask] = [feed_examples[j].question_attention_mask for j in range(batch_size)]

    feed_dict[gr.batch_answer] = [feed_examples[j].answer for j in range(batch_size)]

    feed_dict[gr.batch_number_column] = [feed_examples[j].columns for j in range(batch_size)]

    feed_dict[gr.batch_processed_number_column] = [feed_examples[j].processed_number_columns for j in range(batch_size)]

    feed_dict[gr.batch_processed_sorted_index_number_column] = [feed_examples[j].sorted_number_index for j in range(batch_size)]

    feed_dict[gr.batch_processed_sorted_index_word_column] = [feed_examples[j].sorted_word_index for j in range(batch_size)]

    feed_dict[gr.batch_question_number] = np.array([feed_examples[j].question_number for j in range(batch_size)]).reshape((batch_size, 1))

    feed_dict[gr.batch_question_number_one] = np.array([feed_examples[j].question_number_1 for j in range(batch_size)]).reshape((batch_size, 1))

    feed_dict[gr.batch_question_number_mask] = [feed_examples[j].question_number_mask for j in range(batch_size)]

    feed_dict[gr.batch_question_number_one_mask] = np.array([feed_examples[j].question_number_one_mask for j in range(batch_size)]).reshape((batch_size, 1))

    feed_dict[gr.batch_print_answer] = [feed_examples[j].print_answer for j in range(batch_size)]

    feed_dict[gr.batch_exact_match] = [feed_examples[j].exact_match for j in range(batch_size)]

    feed_dict[gr.batch_group_by_max] = [feed_examples[j].group_by_max for j in range(batch_size)]

    feed_dict[gr.batch_column_exact_match] = [feed_examples[j].exact_column_match for j in range(batch_size)]

    feed_dict[gr.batch_ordinal_question] = [feed_examples[j].ordinal_question for j in range(batch_size)]

    feed_dict[gr.batch_ordinal_question_one] = [feed_examples[j].ordinal_question_one for j in range(batch_size)]

    feed_dict[gr.batch_number_column_mask] = [feed_examples[j].column_mask for j in range(batch_size)]

    feed_dict[gr.batch_number_column_names] = [feed_examples[j].column_ids for j in range(batch_size)]

    feed_dict[gr.batch_processed_word_column] = [feed_examples[j].processed_word_columns for j in range(batch_size)]

    feed_dict[gr.batch_word_column_mask] = [feed_examples[j].word_column_mask for j in range(batch_size)]

    feed_dict[gr.batch_word_column_names] = [feed_examples[j].word_column_ids for j in range(batch_size)]

    feed_dict[gr.batch_word_column_entry_mask] = [feed_examples[j].word_column_entry_mask for j in range(batch_size)]

    return feed_dict

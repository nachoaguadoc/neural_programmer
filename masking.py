# -*- coding: utf-8 -*-

from __future__ import division
#from para import FLAGS
from random import Random
from utils.helpers import *
from sys import exit
from wikiqa import WikiExample, WikiQuestionLoader, TableInfo, WikiQuestionGenerator
from data_utils import *

import tensorflow as tf
import copy
import numbers
import numpy as np
import wikiqa
import cPickle


seen_tables = {}

def no_number(input_entry):
    return not any(char.isdigit() for char in input_entry)


def np_softmax(input_column):
    ex = np.exp(input_column - np.max(input_column))
    return ex / ex.sum()


def sim_match(e, emb_dict, question, table, number):
    answer = np.zeros(np.asarray(table).shape).tolist()
    match = {}
    for _idx, _ in enumerate(table):
        for idx_, _ in enumerate(table[_idx]):
            for word in question:
                if number:
                    if word == table[_idx][idx_]:
                        answer[_idx][idx_] = 1.
                        match[_idx] = 1.
                else:
                    # print table[_idx][idx_]
                    pass
    if not number and len(e.sim_tokens) > 0:
        for q_wd in e.sim_tokens:
            try:
                q_wd_emb = np.asarray(e.sim_tokens_emb[q_wd])
            except KeyError:
                # print 'The word in question doesnt has embedding from Glove, Skipped.'
                continue
            for _idx, _ in enumerate(table):
                for idx_, _ in enumerate(table[_idx]):
                    for cell_word in table[_idx][idx_]:
                        if cell_word != 'UNK' and isinstance(cell_word, basestring) and no_number(cell_word):
                            try:
                                cell_word_emb = emb_dict[cell_word]
                            except KeyError:
                                cell_word_emb = np.zeros(len(q_wd_emb))
                            answer[_idx][idx_] += np.dot(cell_word_emb, q_wd_emb)
    for idx, column in enumerate(answer):
        answer[idx][:] = np_softmax(column)
    return answer, match


def partial_match(question, table, number):
    answer = []
    match = {}
    for i in range(len(table)):
        temp = []
        for j in range(len(table[i])):
            temp.append(0)
        answer.append(temp)
    for i in range(len(table)):
        for j in range(len(table[i])):
            for word in question:
                if (number):
                    if (word == table[i][j]):
                        answer[i][j] = 1.0
                        match[i] = 1.0
                else:
                    if (word in table[i][j]):
                        answer[i][j] = 1.0
                        match[i] = 1.0
    return answer, match


def exact_match(question, table, number):
    #performs exact match operation
    answer = []
    match = {}
    matched_indices = []
    for i in range(len(table)):
        temp = []
        for j in range(len(table[i])):
            temp.append(0)
        answer.append(temp)

    for i in range(len(table)):
        for j in range(len(table[i])):
            if number:
                for word in question:
                    if (word == table[i][j]):
                        match[i] = 1.0
                        answer[i][j] = 1.0
            else:
                table_entry = table[i][j]
                for k in range(len(question)):
                    if (k + len(table_entry) <= len(question)):
                        if (table_entry == question[k:(k + len(table_entry))]):
                            #if(len(table_entry) == 1):
                            #print "match: ", table_entry, question
                            match[i] = 1.0
                            answer[i][j] = 1.0
                            matched_indices.append((k, len(table_entry)))

    return answer, match, matched_indices


def partial_column_match(question, table, number):
    answer = []
    for i in range(len(table)):
        answer.append(0)
    for i in range(len(table)):
        for word in question:
            if (word in table[i]):
                answer[i] = 1.0
    return answer

def partial_column_description_match(question, table, demo):
    answer = []
    add_token = False
    for i in range(len(table)):
        answer.append(0)
    if demo:
        for i in range(len(table)):
            for word in question:
                if word in table[i]:
                    answer[i] = 1.0
                    add_token = True
    return answer, add_token

def exact_column_match(question, table, number):
    #performs exact match on column names
    answer = []
    matched_indices = []
    for i in range(len(table)):
        answer.append(0)
    for i in range(len(table)):
        table_entry = table[i]
        for k in range(len(question)):
            if (k + len(table_entry) <= len(question)):
                if (table_entry == question[k:(k + len(table_entry))]):
                    answer[i] = 1.0
                    matched_indices.append((k, len(table_entry)))
    return answer, matched_indices


def get_max_entry(a):
    e = {}
    for w in a:
        if (w != "UNK, "):
            if (e.has_key(w)):
                e[w] += 1
            else:
                e[w] = 1
    if (len(e) > 0):
        (key, val) = sorted(e.items(), key=lambda x: -1 * x[1])[0]
        if (val > 1):
            return key
        else:
            return -1.0
    else:
        return -1.0


def list_join(a):
    ans = ""
    for w in a:
        ans += str(w) + ", "
    return ans


def group_by_max(table, number):
    #computes the most frequently occurring entry in a column
    answer = []
    for i in range(len(table)):
        temp = []
        for j in range(len(table[i])):
            temp.append(0)
        answer.append(temp)
    for i in range(len(table)):
        if (number):
            curr = table[i]
        else:
            curr = [list_join(w) for w in table[i]]
        max_entry = get_max_entry(curr)
        #print i, max_entry
        for j in range(len(curr)):
            if (max_entry == curr[j]):
                answer[i][j] = 1.0
            else:
                answer[i][j] = 0.0
    return answer


def pick_one(a):
    for i in range(len(a)):
        if (1.0 in a[i]):
            return True
    return False


def check_processed_cols(col, utility):
    return True in [True for y in col if (y != utility.FLAGS.pad_int and y != utility.FLAGS.bad_number_pre_process)]


def complete_wiki_processing(data, utility, key='train'):
    with open('./glove/emb_lookup.pkl', 'r') as f:
        emb_dict = cPickle.load(f)
        print 'Finish loading the glove embedding...'

    train = True if key=='train' else False
    processed_data = []
    num_bad_examples = 0
    for example in data:
        number_found = 0
        if example.is_bad_eg:
            num_bad_examples += 1
        if not example.is_bad_eg:
            example.string_question = example.q[:]
            #entry match
            example.p_nb_cols = example.p_nb_cols[:]
            example.p_wd_cols = example.p_wd_cols[:]

            example.word_exact_match, word_match, matched_indices = exact_match(example.string_question, example.original_wc, number=False)
            example.number_exact_match, number_match, _ = exact_match(example.string_question, example.original_nc, number=True)
            
            if (not (pick_one(example.word_exact_match)) and not (pick_one(example.number_exact_match))):
                assert len(word_match) == 0
                assert len(number_match) == 0
                example.word_exact_match, word_match = partial_match(example.string_question, example.original_wc, number=False)

            if not pick_one(e.word_exact_match) and not pick_one(e.number_exact_match):
                assert len(word_match) == 0
                assert len(number_match) == 0
                example.word_exact_match, word_match = sim_match(example, emb_dict,
                                                              example.string_question,
                                                              example.original_wc,
                                                              number=False)                
            
            #group by max
            example.word_group_by_max = group_by_max(example.original_wc, False)
            example.number_group_by_max = group_by_max(example.original_nc, True)
            #column name match
            example.word_column_exact_match, wcol_matched_indices = exact_column_match(example.string_question, example.original_wc_names, number=False)
            example.number_column_exact_match, ncol_matched_indices = exact_column_match(example.string_question, example.original_nc_names, number=False)
            if (not (1.0 in example.word_column_exact_match) and not (1.0 in example.number_column_exact_match)):
                example.word_column_exact_match = partial_column_match(example.string_question, example.original_wc_names, number=False)
                example.number_column_exact_match = partial_column_match(example.string_question, example.original_nc_names, number=False)
            if (key=='demo' or key=='test'):
                example.word_column_description_match, word_token = partial_column_description_match(example.string_question, example.wd_col_desc, demo=True)
                example.number_column_description_match, col_token = partial_column_description_match(example.string_question, example.nb_col_desc, demo=True)
            else:
                example.word_column_description_match, word_token = partial_column_description_match(example.string_question, example.original_wc_names, demo=False)
                example.number_column_description_match, col_token = partial_column_description_match(example.string_question, example.original_nc_names, demo=False)
            add_token = word_token or col_token
            if (len(word_match) > 0 or len(number_match) > 0):
                example.q.append(utility.entry_match_token)
            if (1.0 in example.word_column_exact_match or 1.0 in example.number_column_exact_match or add_token):
                example.q.append(utility.column_match_token)
            example.string_question = example.q[:]
            example.number_lookup_mat = np.transpose(example.number_lookup_mat)[:]
            example.word_lookup_mat = np.transpose(example.word_lookup_mat)[:]
            example.nb_cols = example.nb_cols[:]
            example.wd_cols = example.wd_cols[:]
            example.len_total_cols = len(example.wd_col_names) + len(example.nb_col_names)
            example.nb_col_names = example.nb_col_names[:]
            example.wd_col_names = example.wd_col_names[:]
            example.nb_col_desc = example.nb_col_desc[:]
            example.wd_col_desc = example.wd_col_desc[:]
            example.string_nb_col_names = example.nb_col_names[:]
            example.string_wd_col_names = example.wd_col_names[:]
            example.sorted_number_index = []
            example.sorted_word_index = []
            example.number_column_mask = []
            example.word_column_mask = []
            example.processed_number_column_mask = []
            example.processed_word_column_mask = []
            example.word_column_entry_mask = []
            example.question_attention_mask = []
            example.question_number = example.question_number_1 = -1
            example.question_attention_mask = []
            example.ordinal_question = []
            example.ordinal_question_one = []
            new_question = []

            if (len(example.nb_cols) > 0):
                example.len_col = len(example.nb_cols[0])
            else:
                example.len_col = len(example.wd_cols[0])
            for (start, length) in matched_indices:
                for j in range(length):
                    example.q[start + j] = utility.unk_token # replace exact matched token with UNK?

            for word in example.q:
                if (isinstance(word, numbers.Number) or wikiqa.is_date(word)):
                    if (not (isinstance(word, numbers.Number)) and wikiqa.is_date(word)):
                        word = word.replace("X", "").replace("-", "")
                    number_found += 1
                    if (number_found == 1):
                        example.question_number = word
                        if (len(example.ordinal_question) > 0):
                            example.ordinal_question[len(example.ordinal_question) - 1] = 1.0
                        else:
                            example.ordinal_question.append(1.0)
                    elif (number_found == 2):
                        example.question_number_1 = word
                        if (len(example.ordinal_question_one) > 0):
                            example.ordinal_question_one[len(example.ordinal_question_one) - 1] = 1.0
                        else:
                            example.ordinal_question_one.append(1.0)
                else:
                    new_question.append(word)
                    example.ordinal_question.append(0.0)
                    example.ordinal_question_one.append(0.0)

            example.q = [utility.word_ids[word_lookup(w, utility)] for w in new_question]
            example.question_attention_mask = [0.0] * len(example.q)
            #when the first question number occurs before a word

            example.ordinal_question = example.ordinal_question[0:len(example.q)]
            example.ordinal_question_one = example.ordinal_question_one[0:len(example.q)]

            #question-padding
            example.q = [utility.word_ids[utility.dummy_token]] * (utility.FLAGS.question_length - len(example.q)) + example.q
            example.question_attention_mask = [-10000.0] * (utility.FLAGS.question_length - len(example.question_attention_mask)) + example.question_attention_mask
            example.ordinal_question = [0.0] * (utility.FLAGS.question_length - len(example.ordinal_question)) + example.ordinal_question
            example.ordinal_question_one = [0.0] * (utility.FLAGS.question_length - len(example.ordinal_question_one)) + example.ordinal_question_one

            if True:
                #number columns and related-padding
                num_cols = len(example.nb_cols)
                start = 0
                for column in example.nb_cols:
                    if (check_processed_cols(example.p_nb_cols[start], utility)):
                        example.processed_number_column_mask.append(0.0)
                    sorted_index = sorted(range(len(example.p_nb_cols[start])), key=lambda k: example.p_nb_cols[start][k], reverse=True)
                    sorted_index = sorted_index + [utility.FLAGS.pad_int] * (utility.FLAGS.max_elements - len(sorted_index))
                    example.sorted_number_index.append(sorted_index)
                    example.nb_cols[start] = column + [utility.FLAGS.pad_int] * (utility.FLAGS.max_elements - len(column))
                    example.p_nb_cols[start] += [utility.FLAGS.pad_int] * (utility.FLAGS.max_elements - len(example.p_nb_cols[start]))
                    start += 1
                    example.number_column_mask.append(0.0)
                for remaining in range(num_cols, utility.FLAGS.max_number_cols):
                    example.sorted_number_index.append([utility.FLAGS.pad_int] * (utility.FLAGS.max_elements))
                    example.nb_cols.append([utility.FLAGS.pad_int] * (utility.FLAGS.max_elements))
                    example.p_nb_cols.append([utility.FLAGS.pad_int] * (utility.FLAGS.max_elements))
                    example.number_exact_match.append([0.0] * (utility.FLAGS.max_elements))
                    example.number_group_by_max.append([0.0] * (utility.FLAGS.max_elements))
                    example.number_column_mask.append(-100000000.0)
                    example.processed_number_column_mask.append(-100000000.0)
                    example.number_column_exact_match.append(0.0)
                    example.number_column_description_match.append(0.0)
                    example.nb_col_names.append([utility.dummy_token])
                    example.nb_col_desc.append([utility.dummy_token] * utility.FLAGS.max_description)

                #word column  and related-padding
                start = 0
                word_num_cols = len(example.wd_cols)
                for column in example.wd_cols:
                    if (check_processed_cols(example.p_wd_cols[start], utility)):
                        example.processed_word_column_mask.append(0.0)
                    sorted_index = sorted(range(len(example.p_wd_cols[start])), key=lambda k: example.p_wd_cols[start][k], reverse=True)
                    sorted_index = sorted_index + [utility.FLAGS.pad_int] * (utility.FLAGS.max_elements - len(sorted_index))
                    example.sorted_word_index.append(sorted_index)
                    column, _ = convert_to_int_2d_and_pad(column, utility, utility.FLAGS.max_entry_length, False)
                    example.wd_cols[start] = column + [[utility.word_ids[utility.dummy_token]] * utility.FLAGS.max_entry_length] * (utility.FLAGS.max_elements - len(column))
                    example.p_wd_cols[start] += [utility.FLAGS.pad_int] * (utility.FLAGS.max_elements - len(example.p_wd_cols[start]))
                    example.word_column_entry_mask.append([0] * len(column) + [utility.word_ids[utility.dummy_token]] * (utility.FLAGS.max_elements - len(column)))
                    start += 1
                    example.word_column_mask.append(0.0)
                for remaining in range(word_num_cols, utility.FLAGS.max_word_cols):
                    example.sorted_word_index.append([utility.FLAGS.pad_int] * (utility.FLAGS.max_elements))
                    example.wd_cols.append([[utility.word_ids[utility.dummy_token]] * utility.FLAGS.max_entry_length] * (utility.FLAGS.max_elements))
                    example.word_column_entry_mask.append([utility.word_ids[utility.dummy_token]] * (utility.FLAGS.max_elements))
                    example.word_exact_match.append([0.0] * (utility.FLAGS.max_elements))
                    example.word_group_by_max.append([0.0] * (utility.FLAGS.max_elements))
                    example.p_wd_cols.append([utility.FLAGS.pad_int] * (utility.FLAGS.max_elements))
                    example.word_column_mask.append(-100000000.0)
                    example.processed_word_column_mask.append(-100000000.0)
                    example.word_column_exact_match.append(0.0)
                    example.word_column_description_match.append(0.0)
                    example.wd_col_names.append([utility.dummy_token] * utility.FLAGS.max_entry_length)
                    example.wd_col_desc.append([utility.dummy_token] * utility.FLAGS.max_description)

                seen_tables[example.tb_key] = 1
            true_mask = [1.] * utility.FLAGS.embedding_dims
            false_mask = [0.] * utility.FLAGS.embedding_dims
            #convert column and word column names to integers
            example.number_column_ids, example.number_column_name_lengths = convert_to_int_2d_and_pad(example.nb_col_names, utility, utility.FLAGS.max_entry_length, True)
            example.number_column_description_ids, example.number_column_description_lengths = convert_to_int_2d_and_pad(example.nb_col_desc, utility, utility.FLAGS.max_description, True)

            example.number_column_name_mask = []
            for ci in example.number_column_ids:
                temp_mask = []
                for id_ in ci:
                    if id_== utility.dummy_token_id or id_==utility.unk_token:
                        temp_mask.append(false_mask)
                    else:
                        temp_mask.append(true_mask)
                example.number_column_name_mask.append(temp_mask)

            example.number_column_description_mask = []
            for ci in example.number_column_description_ids:
                temp_mask = []
                for id_ in ci:
                    if id_== utility.dummy_token_id or id_==utility.unk_token:
                        temp_mask.append(false_mask)
                    else:
                        temp_mask.append(true_mask)
                example.number_column_description_mask.append(temp_mask)

            example.word_column_ids, example.word_column_name_lengths = convert_to_int_2d_and_pad(example.wd_col_names, utility, utility.FLAGS.max_entry_length, True)
            example.word_column_description_ids, example.word_column_description_lengths = convert_to_int_2d_and_pad(example.wd_col_desc, utility, utility.FLAGS.max_description, True)

            example.word_column_name_mask = []
            for ci in example.word_column_ids:
                temp_mask = []
                for id_ in ci:
                    if id_== utility.dummy_token_id or id_==utility.unk_token:
                        temp_mask.append(false_mask)
                    else:
                        temp_mask.append(true_mask)
                example.word_column_name_mask.append(temp_mask)

            example.word_column_description_mask = []
            for ci in example.word_column_description_ids:
                temp_mask = []
                for id_ in ci:
                    if id_== utility.dummy_token_id or id_==utility.unk_token:
                        temp_mask.append(false_mask)
                    else:
                        temp_mask.append(true_mask)
                example.word_column_description_mask.append(temp_mask)

            for i_em in range(len(example.number_exact_match)):
                example.number_exact_match[i_em] = example.number_exact_match[i_em] + [0.0] * (utility.FLAGS.max_elements - len(example.number_exact_match[i_em]))
                example.number_group_by_max[i_em] = example.number_group_by_max[i_em] + [0.0] * (utility.FLAGS.max_elements - len(example.number_group_by_max[i_em]))
            for i_em in range(len(example.word_exact_match)):
                example.word_exact_match[i_em] = example.word_exact_match[i_em] + [0.0] * (utility.FLAGS.max_elements - len(example.word_exact_match[i_em]))
                example.word_group_by_max[i_em] = example.word_group_by_max[i_em] + [0.0] * (utility.FLAGS.max_elements - len(example.word_group_by_max[i_em]))
            example.exact_match = example.number_exact_match + example.word_exact_match
            example.group_by_max = example.number_group_by_max + example.word_group_by_max
            example.exact_column_match = example.number_column_exact_match + example.word_column_exact_match
            example.exact_column_description_match = example.number_column_description_match + example.word_column_description_match
            #answer and related mask, padding
            if ((train and example.is_lookup) or (key=='error-test' and example.is_lookup)):
                example.answer = example.calc_answer
                example.number_print_answer = example.number_lookup_mat.tolist()
                example.word_print_answer = example.word_lookup_mat.tolist()
                for i_answer in range(len(example.number_print_answer)):
                    example.number_print_answer[i_answer] = example.number_print_answer[i_answer] + [0.0] * (utility.FLAGS.max_elements - len(example.number_print_answer[i_answer]))
                for i_answer in range(len(example.word_print_answer)):
                    example.word_print_answer[i_answer] = example.word_print_answer[i_answer] + [0.0] * (utility.FLAGS.max_elements - len(example.word_print_answer[i_answer]))
                example.number_lookup_mat = convert_to_bool_and_pad(example.number_lookup_mat, utility)
                example.word_lookup_mat = convert_to_bool_and_pad(example.word_lookup_mat, utility)
                for remaining in range(num_cols, utility.FLAGS.max_number_cols):
                    example.number_lookup_mat.append([False] * utility.FLAGS.max_elements)
                    example.number_print_answer.append([0.0] * utility.FLAGS.max_elements)
                for remaining in range(word_num_cols, utility.FLAGS.max_word_cols):
                    example.word_lookup_mat.append([False] * utility.FLAGS.max_elements)
                    example.word_print_answer.append([0.0] * utility.FLAGS.max_elements)
                example.print_answer = example.number_print_answer + example.word_print_answer
            elif (train or key=='error-test'):
                example.answer = example.calc_answer
                example.print_answer = [[0.0] * (utility.FLAGS.max_elements)] * (utility.FLAGS.max_number_cols + utility.FLAGS.max_word_cols)
            else:
                example.calc_answer = 0.0
                example.answer = 0.0
                example.print_answer = [[0.0] * (utility.FLAGS.max_elements)] * (utility.FLAGS.max_number_cols + utility.FLAGS.max_word_cols)
            #question_number masks
            if (example.question_number == -1):
                example.question_number_mask = np.zeros([utility.FLAGS.max_elements])
            else:
                example.question_number_mask = np.ones([utility.FLAGS.max_elements])
            if (example.question_number_1 == -1):
                example.question_number_one_mask = -10000.0
            else:
                example.question_number_one_mask = np.float64(0.0)
            if (example.len_col > utility.FLAGS.max_elements):
                continue
            processed_data.append(example)
    return processed_data

if __name__ == '__main__':
    u = Utility()
    u.words = []
    u.word_ids = {}
    u.reverse_word_ids = {}

    train_data = load_from('./p_data'+'/_train.pkl')
    construct_vocab(train_data, u)
    perform_word_cutoff(u)
    add_special_words(u)

    train_data = complete_wiki_processing(train_data, u, True)
    print train_data[0].__dict__.keys()

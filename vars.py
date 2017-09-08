# -*- coding: utf-8 -*-

from __future__ import division
from sys import exit
from utils.helpers import *
from data_utils import *

import numpy as np
import tensorflow as tf


class Parameters(object):
    def __init__(self, u):
        self.utility = u
        self.init_seed_counter = 0
        self.word_init = {}
        # self.params, self.global_step, self.init = self.parameters(u, False)

    def parameters(self, utility, reuse=False):
        embedding_dims = self.utility.FLAGS.embedding_dims
        with tf.variable_scope('collection-vars', reuse=reuse):
            params = {}
            inits = []
            params["unit"] = self.RandomUniformInit([len(utility.operations_set), embedding_dims], name='unit')
            params["word"] = self.RandomUniformInit([utility.FLAGS.vocab_size, embedding_dims], name='wd-embed')
            params["word_match_feature_column_name"] = self.RandomUniformInit([1], name='wmfcon')
            params["controller"] = self.RandomUniformInit([2 * embedding_dims, embedding_dims], name='controller')
            params["column_controller"] = self.RandomUniformInit([2 * embedding_dims, embedding_dims], name='column_controller')
            params["column_controller_prev"] = self.RandomUniformInit([embedding_dims, embedding_dims], name='column_controller_prev')
            params["controller_prev"] = self.RandomUniformInit([embedding_dims, embedding_dims], name='controller_prev')
            global_step = tf.Variable(1, name="global_step")
            #weigths of question and history RNN (or LSTM)
            key_list = ["question_lstm"]
            for key in key_list:
                # Weights going from inputs to nodes.
                for wgts in ["ix", "fx", "cx", "ox"]:
                    params[key + "_" + wgts] = self.RandomUniformInit([embedding_dims, embedding_dims], name=key + "_" + wgts)
                # Weights going from nodes to nodes.
                for wgts in ["im", "fm", "cm", "om"]:
                    params[key + "_" + wgts] = self.RandomUniformInit([embedding_dims, embedding_dims], name=key + "_" + wgts)
                #Biases for the gates and cell
                for bias in ["i", "f", "c", "o"]:
                    if (bias == "f"):
                        print "forget gate bias"
                        params[key + "_" + bias] = tf.random_uniform([embedding_dims], 1.0, 1.1, self.utility.tf_data_type[self.utility.FLAGS.data_type], name=key + "_" + bias)
                    else:
                        params[key + "_" + bias] = self.RandomUniformInit([embedding_dims], name=key + "_" + bias)
            params["history_recurrent"] = self.RandomUniformInit([3 * embedding_dims, embedding_dims], name='history_recurrent')
            params["history_recurrent_bias"] = self.RandomUniformInit([1, embedding_dims], name='history_recurrent_bias')
            params["break_conditional"] = self.RandomUniformInit([2 * embedding_dims, embedding_dims], name='break_conditional')
            init = tf.global_variables_initializer()
        return params, global_step, init

    def RandomUniformInit(self, shape, name):
        """Returns a RandomUniform Tensor between -param_init and param_init."""
        param_seed = self.utility.FLAGS.param_seed
        self.init_seed_counter += 1
        return tf.random_uniform(
            shape, -1.0 *
            (np.float32(self.utility.FLAGS.param_init)
            ).astype(self.utility.np_data_type[self.utility.FLAGS.data_type]),
            (np.float32(self.utility.FLAGS.param_init)
            ).astype(self.utility.np_data_type[self.utility.FLAGS.data_type]),
            self.utility.tf_data_type[self.utility.FLAGS.data_type],
            param_seed + self.init_seed_counter, name=name)


if __name__ == '__main__':
    u = Utility()
    u.words = []
    u.word_ids = {}
    u.reverse_word_ids = {}
    obj = Parameters(u)
    # print obj.params
    # print obj.global_step
    # print obj.init

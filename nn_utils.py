# -*- coding: utf-8 -*-

import tensorflow as tf

def get_embedding(word, utility, params):
    return tf.nn.embedding_lookup(params["word"], word)


def apply_dropout(x, dropout_rate, mode):
    if (dropout_rate > 0.0):
        if (mode == "train"):
            x = tf.nn.dropout(x, dropout_rate)
        else:
            x = x
    return x


def LSTMCell(x, mprev, cprev, key, params):
    """
    Args:
    w: A dictionary of the weights and optional biases as returned
      by LSTMParametersSplit().
    x: Inputs to this cell.
    mprev: m_{t-1}, the recurrent activations (same as the output) from the previous cell.
    cprev: c_{t-1}, the cell activations from the previous cell.
    keep_prob: Keep probability on the input and the outputs of a cell.

    Returns:
    m: Outputs of this cell.
    c: Cell Activations.
    """

    i = tf.matmul(x, params[key + "_ix"]) + tf.matmul(mprev, params[key + "_im"])
    i = tf.nn.bias_add(i, params[key + "_i"])
    f = tf.matmul(x, params[key + "_fx"]) + tf.matmul(mprev, params[key + "_fm"])
    f = tf.nn.bias_add(f, params[key + "_f"])
    c = tf.matmul(x, params[key + "_cx"]) + tf.matmul(mprev, params[key + "_cm"])
    c = tf.nn.bias_add(c, params[key + "_c"])
    o = tf.matmul(x, params[key + "_ox"]) + tf.matmul(mprev, params[key + "_om"])
    o = tf.nn.bias_add(o, params[key + "_o"])
    i = tf.sigmoid(i, name="i_gate")
    f = tf.sigmoid(f, name="f_gate")
    o = tf.sigmoid(o, name="o_gate")
    c = f * cprev + i * tf.tanh(c)
    m = o * c
    return m, c

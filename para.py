# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
# Define the parameters for the preprocessing and running the model

tf.flags.DEFINE_integer("train_steps", 100000, "Number of steps to train")
tf.flags.DEFINE_integer("eval_cycle", 500, "Evaluate model every N steps")
tf.flags.DEFINE_integer("write_every", 500, "Write a checkpoint every N steps")

tf.flags.DEFINE_string("mode", "train", "Mode of execution: train/demo-visual/demo-console/error-test")
tf.flags.DEFINE_string("job_id", "_baseline", "Name of the model trained")
tf.flags.DEFINE_string("model", "baseline", "Eval with baseline/column_description")
tf.flags.DEFINE_string("output_dir", "model/embeddings/", "Path where the trained model will be saved")
tf.flags.DEFINE_string("model_id", "96500", "ID of the checkpoint to retrieve for testing")
tf.flags.DEFINE_string("data_dir", "data/", "Path where the data is stored")

tf.flags.DEFINE_integer("max_elements", 100, "Maximum rows that are considered for processing")
tf.flags.DEFINE_integer("max_description", 100, "Maximum words that are considered for the description")
tf.flags.DEFINE_integer("max_number_cols", 15, "Maximum number columns that are considered for processing")
tf.flags.DEFINE_integer("max_word_cols", 25, "Maximum number columns that are considered for processing")
tf.flags.DEFINE_integer("question_length", 62, "Maximum length of the question")
tf.flags.DEFINE_integer("max_entry_length", 3, "Maximum length of the content of a cell in a table")

tf.flags.DEFINE_string("data_type", "double", "float or double")
tf.flags.DEFINE_float("pad_int", -20000.0, "Padding number used for number columns")
tf.flags.DEFINE_float("bad_number_pre_process", -200000.0, "Number used for corrupted table entry in a number column")

tf.flags.DEFINE_string("word_embeddings", "glove", "custom/glove/word2vecfasttext")
tf.flags.DEFINE_string("embeddings_file", "word_embeddings/glove/glove.6B.50d.txt", "path of the embeddings file")
tf.flags.DEFINE_string("embeddings_pickle", "word_embeddings/glove/glove_50.pkl", "path of the embeddings file")
tf.flags.DEFINE_integer("embedding_dims", 256, "Dimensions of the word embeddings")
tf.flags.DEFINE_integer("vocab_size", 10800, "Maximum size of the vocabulary")

tf.flags.DEFINE_integer("max_passes", 2, "Number of steps the model executes")
tf.flags.DEFINE_integer("batch_size", 20, "Size of the batch")
tf.flags.DEFINE_float("certainty_threshold", 70.0, "During demo, any answer below this threshold will not be used")

tf.flags.DEFINE_float("clip_gradients", 1.0, "")
tf.flags.DEFINE_float("eps", 1e-6, "")
tf.flags.DEFINE_float("param_init", 0.1, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("l2_regularizer", 0.0001, "")
tf.flags.DEFINE_float("print_cost", 50.0, "Weighting factor in the objective function")
tf.flags.DEFINE_float("dropout", 0.8, "Dropout keep probability")
tf.flags.DEFINE_float("rnn_dropout", 0.9, "Dropout keep probability for rnn connections")
tf.flags.DEFINE_float("word_dropout_prob", 0.9, "Word dropout keep probability")
tf.flags.DEFINE_integer("word_cutoff", 10, "")

tf.flags.DEFINE_integer("param_seed", 150, "")
tf.flags.DEFINE_integer("python_seed", 200, "")
tf.flags.DEFINE_float("max_math_error", 3.0, "max square loss error that is considered")
tf.flags.DEFINE_float("soft_min_value", 5.0, "")

FLAGS = tf.flags.FLAGS

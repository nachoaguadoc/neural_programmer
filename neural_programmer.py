# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Neural Programmer model described in https://openreview.net/pdf?id=ry2YOrcge

This file calls functions to load & pre-process data, construct the TF graph
and performs training or evaluation as specified by the flag evaluator_job
Author: aneelakantan (Arvind Neelakantan)
"""
import time
from random import Random
import numpy as np
import tensorflow as tf
import model
import wiki_data
import parameters
import data_utils
import socket 
import config

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


class Utility:
    #holds FLAGS and other variables that are used in different files
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
        self.operations_set = ["count"] + [
            "prev", "next", "first_rs", "last_rs", "group_by_max", "greater",
            "lesser", "geq", "leq", "max", "min", "word-match"
        ] + ["reset_select"] + ["print"]
        self.word_ids = {}
        self.reverse_word_ids = {}
        self.word_count = {}
        self.random = Random(FLAGS.python_seed)


def main(args):
    utility = Utility()
    train_name = "random-split-1-train.examples"
    dev_name = "random-split-1-dev.examples"
    test_name = "pristine-unseen-tables.examples"

    #Load the training, validation and test data
    dat = wiki_data.WikiQuestionGenerator(train_name, dev_name, test_name, FLAGS.data_dir)
    train_data, dev_data, test_data = dat.load(FLAGS.mode, FLAGS.model)

    # Construct the vocabulary 
    utility.words = []
    utility.word_ids = {}
    utility.reverse_word_ids = {}

    data_utils.construct_vocab(train_data, utility)
    data_utils.construct_vocab(dev_data, utility, True)
    data_utils.construct_vocab(test_data, utility, True)
    data_utils.add_special_words(utility)
    #data_utils.perform_word_cutoff(utility)


    train_data = data_utils.complete_wiki_processing(train_data, utility, 'train')
    dev_data = data_utils.complete_wiki_processing(dev_data, utility, 'error-test')
    #test_data = data_utils.complete_wiki_processing(test_data, utility, False)

    print("Preprocessing finished:")
    print("     Number of train examples ", len(train_data))
    print("     Number of validation examples ", len(dev_data))
    print("     Number of test examples ", len(test_data))

    #construct TF graph and train or evaluate
    master(train_data, dev_data, utility, dat)


def master(train_data, dev_data, utility, dat):

    #Initialize the parameters of the model
    param_class = parameters.Parameters(utility)
    params, global_step, init = param_class.parameters(utility)

    batch_size = utility.FLAGS.batch_size 
    model_dir = utility.FLAGS.output_dir + "/model_" + utility.FLAGS.job_id + "/"
    model_file = 'model_' + utility.FLAGS.model_id

    key = utility.FLAGS.mode

    print("Running model in mode ", key)


    graph = model.Graph(utility, batch_size, utility.FLAGS.max_passes, mode=key)
    graph.create_graph(params, global_step)

    #start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init.name)
        sess.run(graph.init_op.name)
        to_save = params.copy()
        saver = tf.train.Saver(to_save, max_to_keep=500)

        if key == 'custom-test':
            print("restoring: ", model_file)
            saver.restore(sess, model_dir + model_file)

            custom_test(graph, utility, batch_size, sess, model_dir, dat, 'data/custom/uefa.examples')

        elif key == 'error-test':
            selected_models = {}
            file_list = tf.gfile.ListDirectory(model_dir)

            for model_file in file_list:
                if "data" in model_file:
                    model_file = model_file.split(".")[0]
                else:
                    continue
                model_step = int(model_file.split("_")[len(model_file.split("_")) - 1])
                selected_models[model_step] = model_file

            file_list = sorted(selected_models.items(), key=lambda x: x[0])

            print "List of models to be evaluated: ", file_list

            testing_accuracy = []
            for model_file in file_list:
                model_file = model_file[1]
                print "Restoring ", model_file
                saver.restore(sess, model_dir + model_file)
                model_step = int(model_file.split("_")[len(model_file.split("_")) - 1])

                print "Evaluating model ", model_file, model_step
                accuracy = test(sess, dev_data, batch_size, graph, model_step)
                testing_accuracy.append(accuracy)

            text_file = open(model_dir + "testing_accuracy.txt", "w")
            text_file.write(str(testing_accuracy))
            text_file.close()

        elif key == 'train':
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if not (tf.gfile.IsDirectory(utility.FLAGS.output_dir)):
                print "Creating output directory ", utility.FLAGS.output_dir
                tf.gfile.MkDir(utility.FLAGS.output_dir)
            if not (tf.gfile.IsDirectory(model_dir)):
                print "Creating model directory ", model_dir
                tf.gfile.MkDir(model_dir)
            train(graph, utility, batch_size, train_data, sess, model_dir, saver)

        elif key == 'demo-visual':
            print("Restoring ", model_file)
            saver.restore(sess, model_dir + model_file)
            demo(graph, utility, sess, model_dir, dat, 'visual')

        elif key == 'demo-console':
            demo(graph, utility, sess, model_dir, dat, 'console')


# Evaluate the accuracy of the model with a set of given questions as input
def custom_test(graph, utility, batch_size, sess, model_dir, dat, file_name):

    ids, questions, table_keys, answers = wiki_data.load_custom_questions(file_name)
    data = []
    for i in range(len(questions)):
        example = dat.load_example(ids[i], questions[i], table_keys[i])
        data.append(example) 

    data_utils.construct_vocab(data, utility, True)
    final_data = data_utils.complete_wiki_processing(data, utility, 'demo')
    predictions = predict(sess, final_data, answers, batch_size, graph, table_keys[0], dat)
    total = len(predictions)

    correct = 0.0
    for i in range(total):
        if predictions[i] == answers[i]:
            correct += 1
        else:
            print(questions[i], predictions[i], answers[i])
    accuracy = (correct / total) * 100
    print("Total test cases:", total)
    print("Correct answers:", correct)
    print("Accuracy:", accuracy)

def predict(sess, data, answers, batch_size, graph, table_key, dat):

    final_predictions = []
    for curr in range(0, len(data) - batch_size + 1, batch_size):
        [predictions] = sess.run([graph.answers], feed_dict=data_utils.generate_feed_dict(data, curr, batch_size, graph))

    for i in range(batch_size):
        scalar_answer = predictions[0][i]
        lookup_answer = predictions[1][i]

        return_scalar = True
        retrieval_answers = []

        for col in range(len(lookup_answer)):
            if not all(p == 0 for p in lookup_answer[col]):
                return_scalar = False
            retrieval_answers.append([[i for i, e in enumerate(lookup_answer[col]) if e != 0], col])

        if return_scalar:
            final_predictions.append(str(int(scalar_answer)))
        else:
            lookup_answer = retrieval_answers[0]
            rows = lookup_answer[0]
            col = lookup_answer[1]
            rows_answer = []

            for row in rows:
                row_answer = ''
                col_index = col if col<15 else col-15
                list_answer = dat.custom_tables[table_key].number_columns[col_index][row]

                if type(list_answer) == float:
                    row_answer = str(int(list_answer))
                else:
                    for l in list_answer:
                        row_answer += " " + str(l)
                    rows_answer.append(row_answer)

            final_answer = ','.join(rows_answer)

        final_answer = final_answer[1:] if final_answer[0] == ' ' else final_answer
        final_predictions.append(final_answer)

    return final_predictions

# Evaluate the accuracy of the model with the given data (normally validation set)
def test(sess, data, batch_size, graph, i):
    num_examples = 0.0
    gc = 0.0

    for j in range(0, len(data) - batch_size + 1, batch_size):
        [ct] = sess.run([graph.final_correct], feed_dict=data_utils.generate_feed_dict(data, j, batch_size, graph))
    gc += ct * batch_size
    num_examples += batch_size
    accuracy = gc / num_examples
    print "Accuracy after ", i, " iterations: ", accuracy
    return accuracy

# Train the model 
def train(graph, utility, batch_size, train_data, sess, model_dir,
          saver):
    curr = 0
    train_set_loss = 0.0
    utility.random.shuffle(train_data)
    start = time.time()
    training_loss = []
    for i in range(utility.FLAGS.train_steps):
        curr_step = i
        if (i > 0 and i % FLAGS.write_every == 0):
            model_file = model_dir + "model_" + str(i)
            saver.save(sess, model_file)
        if curr + batch_size >= len(train_data):
            curr = 0
            utility.random.shuffle(train_data)
        step, cost_value = sess.run([graph.step, graph.total_cost], feed_dict=data_utils.generate_feed_dict(train_data, curr, batch_size, graph, train=True, utility=utility))
        curr = curr + batch_size
        train_set_loss += cost_value
        if (i > 0 and i % FLAGS.eval_cycle == 0):
            end = time.time()
            time_taken = end - start
            print("Step ", i, " . Time: ", time_taken, " seconds")
            start = end
            print("Training Loss: ", train_set_loss / utility.FLAGS.eval_cycle)
            training_loss.append(train_set_loss / utility.FLAGS.eval_cycle)
            text_file = open(model_dir + "training_loss.txt", "w")
            text_file.write(str(training_loss))
            text_file.close()
            train_set_loss = 0.0
            eta = (((utility.FLAGS.train_steps - i) / FLAGS.eval_cycle) * time_taken)
            m, s = divmod(eta, 60)
            h, m = divmod(m, 60)
            print "%d:%02d:%02d" % (h, m, s)
            print("Estimated Remaining Time: ", h, " hours, ", m, " minutes")

def demo(graph, utility, sess, model_dir, dat, mode):
    if mode=='visual':
        i = 0
        # Listen to incoming questions
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((config.socket_address, config.socket_port))
        s.listen(1)
        print("Listening to incoming questions...")
        while (True):
            # New question received
            conn, addr = s.accept()
            data = conn.recv(1024).decode("utf-8").split("****----****")

            # Read the data
            table_key = data[0]
            tokens = data[1]
            question_id = 'iac-' + str(i)
            print("Question:", tokens, "Table:", table_key)

            # Load the question into the model 
            data = [dat.load_example(question_id, tokens, table_key)]
            data_utils.construct_vocab(data, utility, True)
            final_data = data_utils.complete_wiki_processing(data, utility, 'demo')

            # Run the model to get an answer
            final_answer, debugging = get_prediction(sess, final_data, graph, utility)

            certainty = debugging['certainty']
            if (certainty < FLAGS.certainty_threshold):
                final_answer = "I cannot answer that question with the information in the table."

            print("Answer:", final_answer + "\n")
    
            result = {"answer": final_answer, "debugging": debugging}
            result = str(result)
            i += 1
            conn.send(result.encode())
            conn.close()

    elif mode=='console':
        i = 0
        print("Listening to incoming questions...")

        while (True):
            question_id = 'iac-' + str(i)
            table_key = raw_input("> What table do you want? \n")
            table_key = "csv/custom-csv/" + table_key + ".csv"
            while (True):
                tokens = raw_input("> ")
                print("\n")
                if tokens == 'new':
                    break
                print("Question:", tokens, "Table:", table_key)
                data = [dat.load_example(question_id, tokens, table_key)]
                data_utils.construct_vocab(data, utility, True)
                final_data = data_utils.complete_wiki_processing(data, utility, 'demo')

                final_answer, debugging = get_prediction(sess, final_data, graph, utility)

                certainty = debugging['certainty']

                if (certainty < FLAGS.certainty_threshold):
                    print("> I do not know the answer to your question, although I would say..." + "\n")
                print "> " + final_answer + "\n"
                i += 1

def get_prediction(sess, data, graph, utility):

    debugging = get_steps(sess, data, graph, utility)

    answers = sess.run([graph.answers], feed_dict=data_utils.generate_feed_dict(data, curr, batch_size, graph))[0]
    scalar_answer = answers[0][0]
    lookup_answer = answers[1][0]

    lookup_answers = []
    j = 0
    for col in range(len(lookup_answer)):
        if not all(p == 0 for p in lookup_answer[col]):
            col_index = col if col<15 else col-15
            col_real_index = data[0].number_column_indices[col_index]
            col_name = data[j].number_column_names[col_index]

            rows = [i for i, e in enumerate(lookup_answer[col]) if e != 0]
            for r in rows:
                debugging['cells_answer_neural'].append([r, col_index])
        
            rows_answer = []
            for row in rows:
                row_answer = ''
                col = col-15 if col>=15 else col
                list_answer = dat.custom_tables[table_key].number_columns[col][row]
                if type(list_answer) == float:
                    debugging['answer_neural'].append(list_answer)
                    row_answer = str(list_answer)
                else:
                    row_answer = " ".join([str(i) for i in list_answer])
                    debugging['answer_neural'].append(row_answer)
                rows_answer.append(row_answer)
            debugging['is_lookup_neural'] = True
            lookup_answer = ','.join(rows_answer)
            return (lookup_answer, debugging)

    debugging['is_lookup_neural'] = False
    debugging['answer_neural'].append(int(scalar_answer))
    return (str(scalar_answer), debugging)
    
def get_steps(sess, data, graph, utility):
    debugging =  {
        'question': '',
        'table_key': '',
        'correct': True,
        'threshold': FLAGS.certainty_threshold,
        'steps': [],
        'answer_neural': [],
        'cells_answer_neural': [],
        'is_lookup_neural': True,
        'answer_feedback': [],
        'cells_answer_feedback': [],
        'is_lookup_feedback': True,
        'below_threshold': False,
        'certainty': 0.0
    }

    steps = sess.run([graph.steps], feed_dict=data_utils.generate_feed_dict(data, curr, 1, graph))[0]
    ops = steps['ops']
    cols = steps['cols']
    rows = steps['rows']
    soft_ops = steps['soft_ops']
    soft_cols = steps['soft_cols']
    certainty = 0
    
    print("-------------- New question --------------")
    # Debugging step by step
    for i in range(len(ops)):
        step =  {
          'index': i,
          'operation_index': 0,
          'operation_name': '',
          'operation_softmax': 0,
          'column_index': 0,
          'column_name': '',
          'column_softmax': 0,
          'rows': [],
          'correct': True
        }

        step['operation_index'] = np.where(ops[i] == 1)[1][0]
        step['operation_name'] = utility.operations_set[step['operation_index']]
        step['operation_softmax'] = soft_ops[i][0][step['operation_index']]

        col_index = np.where(cols[i] == 1)[1][0]
        col_real_index = col_index if col_index < 15 else col_index-15
        col = data[0].number_column_names[col_real_index]
        step['column_index'] = data[0].number_column_indices[col_real_index]

        step['column_name'] = " ".join([str(j) for j in col if j!="dummy_token"])
        step['column_softmax'] = soft_cols[i][0][col_index]
       
        step['rows'] =  np.ndarray.tolist(np.where(rows[i] == 1)[1])

        debugging['steps'].append(step)

        certainty_step = step['operation_softmax'] * step['column_softmax']
        certainty += certainty_step

        print("STEP ", str(i), " : Performed operation <", step['operation_name'], "> (", + str(step['operation_softmax']) + ") over the column <", step['column_name'], "> (" + str(step['column_softmax']) + ")")

    certainty = (certainty / len(ops)) * 100
    debugging['certainty'] = certainty
    print("Final confidence: " + str(certainty))
    if (certainty < FLAGS.certainty_threshold):
        debugging['below_threshold'] = True
        print("(Below threshold of ", str(FLAGS.certainty_threshold), "%")
    print("-------------------------------------------")

    return debugging

if __name__ == "__main__":
  tf.app.run()

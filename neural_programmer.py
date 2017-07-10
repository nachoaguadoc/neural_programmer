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
tf.flags.DEFINE_integer("train_steps", 100001, "Number of steps to train")
tf.flags.DEFINE_integer("eval_cycle", 500, "Evaluate model at every eval_cycle steps")
tf.flags.DEFINE_integer("max_elements", 100, "maximum rows that are  considered for processing")
tf.flags.DEFINE_integer("max_description", 100, "maximum words that are  considered for processing the description")
tf.flags.DEFINE_integer("max_number_cols", 15, "maximum number columns that are considered for processing")
tf.flags.DEFINE_integer("max_word_cols", 25, "maximum number columns that are considered for processing")
tf.flags.DEFINE_integer("question_length", 62, "maximum question length")
tf.flags.DEFINE_integer("max_entry_length", 3, "")
tf.flags.DEFINE_integer("max_passes", 2, "number of operation passes")
tf.flags.DEFINE_integer("embedding_dims", 256, "")
tf.flags.DEFINE_integer("batch_size", 1, "")

tf.flags.DEFINE_float("clip_gradients", 1.0, "")
tf.flags.DEFINE_float("eps", 1e-6, "")
tf.flags.DEFINE_float("param_init", 0.1, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("l2_regularizer", 0.0001, "")
tf.flags.DEFINE_float("print_cost", 50.0, "weighting factor in the objective function")
tf.flags.DEFINE_float("certainty_threshold", 70.0, "")

tf.flags.DEFINE_string("mode", "demo", """mode""")
tf.flags.DEFINE_string("job_id", "_baseline", """job id""")
tf.flags.DEFINE_string("model", "baseline", """model to evaluate""")
tf.flags.DEFINE_string("output_dir", "model/embeddings/", """output_dir""")
tf.flags.DEFINE_string("model_id", "96500", """model id""")
tf.flags.DEFINE_string("data_dir", "data/", """data_dir""")

tf.flags.DEFINE_integer("write_every", 500, "write every N")
tf.flags.DEFINE_integer("param_seed", 150, "")
tf.flags.DEFINE_integer("python_seed", 200, "")

tf.flags.DEFINE_float("dropout", 0.8, "dropout keep probability")
tf.flags.DEFINE_float("rnn_dropout", 0.9, "dropout keep probability for rnn connections")
tf.flags.DEFINE_float("pad_int", -20000.0, "number columns are padded with pad_int")
tf.flags.DEFINE_string("data_type", "double", "float or double")
tf.flags.DEFINE_float("word_dropout_prob", 0.9, "word dropout keep prob")
tf.flags.DEFINE_integer("word_cutoff", 10, "")
tf.flags.DEFINE_integer("vocab_size", 10800, "")
tf.flags.DEFINE_string("job_mode", "train", "whether to run as trainer/evaluator/demo")
tf.flags.DEFINE_float("bad_number_pre_process", -200000.0, "number that is added to a corrupted table entry in a number column")
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


def evaluate(sess, data, batch_size, graph, i):
  #computes accuracy
  num_examples = 0.0
  gc = 0.0
  for j in range(0, len(data) - batch_size + 1, batch_size):
    [ct] = sess.run([graph.final_correct],
                    feed_dict=data_utils.generate_feed_dict(data, j, batch_size,
                                                            graph))
    gc += ct * batch_size
    num_examples += batch_size
  print "dev set accuracy   after ", i, " : ", gc / num_examples
  print num_examples, len(data)
  print "--------"

def evaluate_custom(sess, data, answers, batch_size, graph, table_key, dat):
  #computes accuracy
  num_examples = 0.0
  gc = 0.0
  final_predictions = []
  for curr in range(0, len(data) - batch_size + 1, batch_size):
    [predictions] = sess.run([graph.answers], feed_dict=data_utils.generate_feed_dict(data, curr, batch_size, graph))

    for i in range(batch_size):
      scalar_answer = predictions[0][i]
      lookup_answer = predictions[1][i]

      return_scalar = True
      lookup_answers = []
      j = 0
      for col in range(len(lookup_answer)):
        if not all(p == 0 for p in lookup_answer[col]):
          return_scalar = False
          if col < 15:
            col_name = data[j].number_column_names[col]
          else:
            col_name = data[j].word_column_names[col-15]
          lookup_answers.append([[i for i, e in enumerate(lookup_answer[col]) if e != 0], col])

      if return_scalar:
        final_predictions.append(str(int(scalar_answer)))
      else:
        a = lookup_answers[0]
        rows = a[0]
        col = a[1]
        rows_answer = []
        for row in rows:
          row_answer = ''
          if col < 15:
            list_answer = dat.custom_tables[table_key].number_columns[col][row]
          else:
            list_answer = dat.custom_tables[table_key].word_columns[col-15][row]
          if type(list_answer) == float:
            row_answer = str(int(list_answer))
          else:
            for l in list_answer:
              row_answer += " " + str(l)
          rows_answer.append(row_answer)

        final_answer = ','.join(rows_answer)

        if final_answer[0] == ' ':
          final_answer = final_answer[1:]

        final_predictions.append(final_answer)

  return final_predictions

def get_prediction(sess, data, graph, utility, debug=True, curr=0, batch_size=1):

  debugging =  {
    'answer_neural': [],
    'cells_answer_neural': [],
    'is_lookup_neural': True,
    'steps': [],
    'threshold': []
  }

  steps = sess.run([graph.steps], feed_dict=data_utils.generate_feed_dict(data, curr, batch_size, graph))
  ops = steps[0]['ops']
  cols = steps[0]['cols']
  rows = steps[0]['rows']
  soft_ops = steps[0]['soft_ops']
  soft_cols = steps[0]['soft_cols']
  certainty = 0
  print("------------- Debugging step by step -------------")
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

    op_index = np.where(ops[i] == 1)[1][0]
    op_name = utility.operations_set[op_index]
    op_certainty = soft_ops[i][0][op_index]
    step['operation_index'] = op_index
    step['operation_name'] = op_name
    step['operation_softmax'] = op_certainty

    col_index = np.where(cols[i] == 1)[1][0]
    if col_index < 15:
      col = data[0].number_column_names[col_index]
      step['column_index'] = col_index
    else:
      col = data[0].word_column_names[col_index-15]
      step['column_index'] = col_index - 15

    col_name = ""
    for c in col:
      if c !='dummy_token':
        col_name += c + " "

    col_certainty = soft_cols[i][0][col_index]

    step['column_name'] = col_name[:-1]
    step['column_softmax'] = col_certainty
    
    row_index =  np.ndarray.tolist(np.where(rows[i] == 1)[1])
    step['rows'] = row_index
    debugging['steps'].append(step)

    certainty_step = op_certainty * col_certainty
    certainty += certainty_step
    print("Certainty step: " + str(certainty_step) + " with cols: " + str(col_certainty) + " certainty ops: " + str(op_certainty))
    print("Step" + str(i) + ": Operation " + op_name + ", Column " + col_name + " and Rows: ", row_index)
  certainty = (certainty / len(ops)) * 100
  print("CERTAINTY: " + str(certainty))
  print("---------------------------------------")

  answers = sess.run([graph.answers], feed_dict=data_utils.generate_feed_dict(data, curr, batch_size, graph))
  scalar_answer = answers[0][0][0]
  lookup_answer = answers[0][1][0]
  print("Scalar output:", scalar_answer)
  print("Lookup output:")
  return_scalar = True
  lookup_answers = []
  j = 0
  for col in range(len(lookup_answer)):
    if not all(p == 0 for p in lookup_answer[col]):
      return_scalar = False
      if col < 15:
        col_index = col
        col_name = data[j].number_column_names[col]
      else:
        col_index = col-15
        col_name = data[j].word_column_names[col-15]

      rows = [i for i, e in enumerate(lookup_answer[col]) if e != 0]
      for r in rows:
        debugging['cells_answer_neural'].append([r, col_index])
      lookup_answers.append([col_name, [i for i, e in enumerate(lookup_answer[col]) if e != 0], col])
      #print("Column name:", col_name, ", Selection;", [i for i, e in enumerate(lookup_answer[col]) if e != 0])

  if return_scalar:
    debugging['is_lookup_neural'] = False
    return ([scalar_answer, debugging], 'scalar', certainty)
  else:
    debugging['is_lookup_neural'] = True
    return ([lookup_answers, debugging], 'lookup', certainty)


def Train(graph, utility, batch_size, train_data, sess, model_dir,
          saver):
  #performs   
  curr = 0
  train_set_loss = 0.0
  utility.random.shuffle(train_data)
  start = time.time()
  for i in range(utility.FLAGS.train_steps):
    curr_step = i
    if (i > 0 and i % FLAGS.write_every == 0):
      model_file = model_dir + "model_" + str(i)
      saver.save(sess, model_file)
    if curr + batch_size >= len(train_data):
      curr = 0
      utility.random.shuffle(train_data)
    step, cost_value = sess.run(
        [graph.step, graph.total_cost],
        feed_dict=data_utils.generate_feed_dict(
            train_data, curr, batch_size, graph, train=True, utility=utility))
    curr = curr + batch_size
    train_set_loss += cost_value
    if (i > 0 and i % FLAGS.eval_cycle == 0):
      end = time.time()
      time_taken = end - start
      print("step ", i, " ", time_taken, " seconds ")
      start = end
      print(" printing train set loss: ", train_set_loss / utility.FLAGS.eval_cycle)
      train_set_loss = 0.0

def Demo(graph, utility, sess, model_dir, dat):
  i = 0
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((config.socket_address, config.socket_port))
  s.listen(1)
  print("Listening to incoming questions...")
  while (True):
    conn, addr = s.accept()
    data = conn.recv(1024).decode("utf-8").split("****----****")
    table_key = data[0]
    tokens = data[1]
    question_id = 'iac-' + str(i)
    print("Question:", tokens, "Table:", table_key)
    example = dat.load_example(question_id, tokens, table_key)
    data = [example] 
    data_utils.construct_vocab(data, utility, True)
    final_data = data_utils.complete_wiki_processing(data, utility, 'demo')
    answer = get_prediction(sess, final_data, graph, utility)
    final_answer = ''

    certainty = answer[2]

    if answer[1] == 'scalar':
      final_answer = str(answer[0][0])
      debugging = str(answer[0][1])
    else:
      print(answer)
      a = answer[0][0][0]
      debugging = answer[0][1]
      rows = a[1]
      col = a[2]
      rows_answer = []
      for row in rows:
        row_answer = ''
        if col < 15:
          list_answer = dat.custom_tables[table_key].number_columns[col][row]
        else:
          list_answer = dat.custom_tables[table_key].word_columns[col-15][row]
        if type(list_answer) == float:
          debugging['answer_neural'].append(list_answer)
          row_answer = str(list_answer)
        else:
          for l in list_answer:
            row_answer += str(l) + " "
          debugging['answer_neural'].append(row_answer[:-1])
        rows_answer.append(row_answer)
       
      final_answer = ','.join(rows_answer)

    print("Answer:", final_answer + "\n")
    debugging['threshold'].append(FLAGS.certainty_threshold)
    if (certainty < FLAGS.certainty_threshold):
      print("I do not know the answer to your question, although that would be my guess.")
      final_answer = "I cannot answer that question with the information in the table."

    result = {"answer": final_answer, "debugging": debugging}
    result = str(result)
    i += 1
    conn.send(result.encode())
    conn.close() 

def DemoConsole(graph, utility, sess, model_dir, dat):
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
      example = dat.load_example(question_id, tokens, table_key)
      data = [example]
      data_utils.construct_vocab(data, utility, True)
      final_data = data_utils.complete_wiki_processing(data, utility, 'demo')
      answer = get_prediction(sess, final_data, graph, utility)
      final_answer = ''

      certainty = answer[2]

      if answer[1] == 'scalar':
        final_answer = str(answer[0][0])
        debugging = str(answer[0][1])
      else:
        print(answer)
        a = answer[0][0][0]
        row = a[1][0]
        col = a[2]
        if col < 15:
          list_answer = dat.custom_tables[table_key].number_columns[col][row]
        else:
          list_answer = dat.custom_tables[table_key].word_columns[col-15][row]
        if type(list_answer) == float:
          final_answer = str(list_answer)
        else:
          for l in list_answer:
            final_answer += " " + str(l)

      print("\n")
      if (certainty < FLAGS.certainty_threshold):
        print("> I do not know the answer to your question, although I would say..." + "\n")
      print "> " + final_answer + "\n"
      i += 1

def Test(graph, utility, batch_size, sess, model_dir, dat, file_name):

    ids, questions, table_keys, answers = wiki_data.load_custom_questions(file_name)
    data = []
    for i in range(len(questions)):
      example = dat.load_example(ids[i], questions[i], table_keys[i])
      data.append(example) 
    
    data_utils.construct_vocab(data, utility, True)
    final_data = data_utils.complete_wiki_processing(data, utility, 'demo')
    predictions = evaluate_custom(sess, final_data, answers, batch_size, graph, table_keys[0], dat)
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

def master(train_data, dev_data, utility, dat):
  #creates TF graph and calls trainer or evaluator
  batch_size = utility.FLAGS.batch_size 
  model_dir = utility.FLAGS.output_dir + "/model_" + utility.FLAGS.job_id + "/"
  #create all paramters of the model
  param_class = parameters.Parameters(utility)
  params, global_step, init = param_class.parameters(utility)
  key = FLAGS.job_mode
  print("Running with key " + key)
  graph = model.Graph(utility, batch_size, utility.FLAGS.max_passes, mode=key)
  graph.create_graph(params, global_step)
  prev_dev_error = 0.0
  final_loss = 0.0
  final_accuracy = 0.0
  #start session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    sess.run(init.name)
    sess.run(graph.init_op.name)
    to_save = params.copy()
    saver = tf.train.Saver(to_save, max_to_keep=500)
    if (key == 'test'):
      model_file = 'model_' + utility.FLAGS.model_id
      print("restoring: ", model_file)
      saver.restore(sess, model_dir + model_file)
      Test(graph, utility, batch_size, sess, model_dir, dat, 'data/custom/uefa.examples')
    if (key == 'error-test'):
      while True:
        selected_models = {}
        file_list = tf.gfile.ListDirectory(model_dir)
        for model_file in file_list:
          if ("checkpoint" in model_file or "index" in model_file or
              "meta" in model_file):
            continue
          if ("data" in model_file):
            model_file = model_file.split(".")[0]
          model_step = int(
              model_file.split("_")[len(model_file.split("_")) - 1])
          selected_models[model_step] = model_file
        file_list = sorted(selected_models.items(), key=lambda x: x[0])
        if (len(file_list) > 0):
          file_list = file_list[0:len(file_list) - 1]
        print "list of models: ", file_list
        for model_file in file_list:
          model_file = model_file[1]
          print "restoring: ", model_file
          saver.restore(sess, model_dir + model_file)
          model_step = int(
              model_file.split("_")[len(model_file.split("_")) - 1])
          print "evaluating on dev ", model_file, model_step
          evaluate(sess, dev_data, batch_size, graph, model_step)

    elif (key == 'train'):
      ckpt = tf.train.get_checkpoint_state(model_dir)
      print "model dir: ", model_dir
      if (not (tf.gfile.IsDirectory(utility.FLAGS.output_dir))):
        print "create dir: ", utility.FLAGS.output_dir
        tf.gfile.MkDir(utility.FLAGS.output_dir)
      if (not (tf.gfile.IsDirectory(model_dir))):
        print "create dir: ", model_dir
        tf.gfile.MkDir(model_dir)
      Train(graph, utility, batch_size, train_data, sess, model_dir,
            saver)
    elif (key == 'demo'):
      #create all paramters of the model
      model_file = 'model_' + utility.FLAGS.model_id
      print("restoring: ", model_file)
      saver.restore(sess, model_dir + model_file)
      if utility.FLAGS.mode == 'console':
        DemoConsole(graph, utility, sess, model_dir, dat)
      else:
        Demo(graph, utility, sess, model_dir, dat)

      
def main(args):
  utility = Utility()
  train_name = "random-split-1-train.examples"
  dev_name = "random-split-1-dev.examples"
  test_name = "pristine-unseen-tables.examples"
  #load data
  dat = wiki_data.WikiQuestionGenerator(train_name, dev_name, test_name, FLAGS.data_dir)
  train_data, dev_data, test_data = dat.load(FLAGS.job_mode, FLAGS.model)
  utility.words = []
  utility.word_ids = {}
  utility.reverse_word_ids = {}
  #construct vocabulary
  data_utils.construct_vocab(train_data, utility)
  data_utils.construct_vocab(dev_data, utility, True)
  data_utils.construct_vocab(test_data, utility, True)
  data_utils.add_special_words(utility)
  data_utils.perform_word_cutoff(utility)
  #convert data to int format and pad the inputs
  train_data = data_utils.complete_wiki_processing(train_data, utility, 'train')
  dev_data = data_utils.complete_wiki_processing(dev_data, utility, 'error-test')

  #test_data = data_utils.complete_wiki_processing(test_data, utility, False)
  print("# train examples ", len(train_data))
  print("# dev examples ", len(dev_data))
  print("# test examples ", len(test_data))
  print("running open source")
  #construct TF graph and train or evaluate
  master(train_data, dev_data, utility, dat)
  
      
if __name__ == "__main__":
  tf.app.run()

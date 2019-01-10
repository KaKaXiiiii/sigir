#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "../data/WikiQA-test.txt", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1507627796/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1507627796/checkpoints/model-1900", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test,x1_temp,x2_temp = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
            batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            #print(batch_predictions)
            all_d = np.concatenate([all_d, batch_sim])
            #print("DEV acc {}".format(batch_acc))

        result = {}
        with open('output.txt', 'w') as outfile:
            for i in range(len(y_test)):
                outfile.write('%s\t%s\t%s\t%s\t%s\n' % (x1_temp[i], x2_temp[i], y_test[i], all_predictions[i], all_d[i]))
                if not result.has_key(x1_temp[i]):
                    result[x1_temp[i]] = []
                result[x1_temp[i]].append((all_predictions[i], y_test[i]))

        rank_all = 0
        count = 0
        for key in result.keys():
            answers = sorted(result[key], key=lambda x:x[0])
            print answers
            rank = 0
            for i in range(len(answers)):
                if answers[i][1] == 1:
                    rank = 1.0/(i+1.0)
                    break
            if rank != 0:
                rank_all += rank
                count +=1

        print 'MRR:' + str(rank_all/count)


        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions))

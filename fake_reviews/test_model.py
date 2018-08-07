from __future__ import print_function

import tensorflow as tf
from six.moves import cPickle as pickle

import util_constants as util

pickle_file = util.TRAIN_DATA_PICKLE
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    test_data = save['test_data']
    test_labels = save['test_labels']
    del save

print("\nEvaluating...\n")

CHECKPOINT_DIR = 'C:/Users/Ajitkumar/Desktop/DK/malware/project/spam_review_tf/runs/1524447709/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        input_size = graph.get_operation_by_name("input_size").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        acc = sess.run(accuracy, {input_x: test_data, input_y: test_labels, dropout_keep_prob: 1.0,
                                  input_size: test_labels.shape[0]})

        print('The test accuracy is ', acc)


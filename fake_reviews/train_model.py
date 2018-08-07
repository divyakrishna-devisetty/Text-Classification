from __future__ import print_function

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import util_constants as util
from model import TF_MODEL



pickle_file = util.VOCAB_PICKLE
with open(pickle_file, 'rb') as f :
    save_file = pickle.load(f)
    word_vectors = save_file['wordsVectors']
    vocab = save_file['vocabulary']
    del save_file

pickle_file = util.TRAIN_DATA_PICKLE
with open(pickle_file, 'rb') as f :
    save_file = pickle.load(f)
    train_data = save_file['train_data']
    train_labels = save_file['train_labels']
    validation_data = save_file['validation_data']
    validation_labels = save_file['validation_labels']
    del save_file

def batch_iteration(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TF_MODEL(sentence_per_review=util.SENTENCE_PER_REVIEW,
                       words_per_sentence=util.WORDS_PER_SENTENCE,
                       wordVectors=word_vectors,
                       embedding_size=util.EMBEDDING_DIM,
                       filter_widths_sent_conv=util.FILTER_WIDTHS_SENT_CONV,
                       num_filters_sent_conv=util.NUM_FILTERS_SENT_CONV,
                       filter_widths_doc_conv=util.FILTER_WIDTHS_DOC_CONV,
                       num_filters_doc_conv=util.NUM_FILTERS_DOC_CONV,
                       num_classes=util.NUM_CLASSES,
                       l2_reg_lambda=util.L2_REG_LAMBDA)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=util.LEARNING_RATE)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=util.NUM_CHECKPOINTS)

        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_size: len(y_batch),
                cnn.dropout: util.DROPOUT_KEEP_PROB
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def run_model(x_batch, y_batch, writer=None):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_size: y_batch.shape[0],
                cnn.dropout: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        print('train data shape ', train_data.shape)
        print('train labels shape ', train_labels.shape)
        print('validation data shape ', validation_data.shape)
        print('validation labels shape ', validation_labels.shape)

        batches = batch_iteration(
            list(zip(train_data, train_labels)), util.BATCH_SIZE, util.NUM_EPOCHS)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % util.EVALUATE_EVERY == 0:
                print("\nEvaluation:")
                run_model(validation_data, validation_labels, writer=dev_summary_writer)
                print("")
            if current_step % util.CHECKPOINT_EVERY == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


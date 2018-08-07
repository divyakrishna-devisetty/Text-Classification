import tensorflow as tf


class TF_MODEL(object):

    
    def __init__(self, sentence_per_review, words_per_sentence, wordVectors, embedding_size, 
                filter_widths_sent_conv, num_filters_sent_conv, filter_widths_doc_conv, num_filters_doc_conv, 
                num_classes, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, shape=(None, sentence_per_review * words_per_sentence), name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=(None, num_classes), name='input_y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.input_size = tf.placeholder(tf.int32, name='input_size')
        
        l2_loss = tf.constant(0.0)
        
        with tf.name_scope('Reshape_Input_X'):
            self.x_reshape = tf.reshape(self.input_x, [self.input_size*sentence_per_review, words_per_sentence])
            self.x_emb = tf.nn.embedding_lookup(wordVectors, self.x_reshape)
            shape = self.x_emb.get_shape().as_list()
            self.x_emb_reshape = tf.reshape(self.x_emb, [self.input_size*sentence_per_review, shape[1], shape[2], 1])
            self.data = tf.cast(self.x_emb_reshape, tf.float32)
            
        conv_outputs = []
        for i, filter_size in enumerate(filter_widths_sent_conv):
            with tf.name_scope('sent_conv-maxpool-tanh-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters_sent_conv]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_sent_conv]), name='b')
                conv = tf.nn.conv2d(
                    self.data,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.bias_add(conv, b)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, words_per_sentence - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                h_output = tf.nn.tanh(pooled, name='tanh')
                conv_outputs.append(h_output)
                
        num_filters_total = num_filters_sent_conv * len(filter_widths_sent_conv)
        self.h_combine = tf.concat(conv_outputs, 3)
        self.h_combine_flat = tf.reshape(self.h_combine, [-1, num_filters_total])
        
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_combine_flat, self.dropout)
        
        self.conv_doc_x = tf.reshape(self.h_drop, [self.input_size, sentence_per_review, num_filters_total])
        self.conv_doc_input = tf.reshape(self.conv_doc_x, [self.input_size, sentence_per_review, num_filters_total, 1])
        
        conv_doc_outputs = []
        for i, filter_size in enumerate(filter_widths_doc_conv):
            with tf.name_scope('doc_conv-maxpool-tanh-%s' % filter_size):
                filter_shape_doc = [filter_size, num_filters_total, 1, num_filters_doc_conv]
                W_doc = tf.Variable(tf.truncated_normal(filter_shape_doc, stddev=0.1), name='W_doc')
                b_doc = tf.Variable(tf.constant(0.1, shape=[num_filters_doc_conv]), name='b_doc')
                conv_doc = tf.nn.conv2d(
                    self.conv_doc_input,
                    W_doc,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv_doc')
                h_doc = tf.nn.bias_add(conv_doc, b_doc)
                pooled_doc = tf.nn.max_pool(
                    h_doc,
                    ksize=[1, sentence_per_review - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool_doc')
                h_output_doc = tf.nn.tanh(pooled_doc, name='tanh')
                conv_doc_outputs.append(h_output_doc)
        
        num_filters_total_doc = num_filters_doc_conv * len(filter_widths_doc_conv)
        self.h_combine_doc = tf.concat(conv_doc_outputs, 3)
        self.h_combine_flat_doc = tf.reshape(self.h_combine_doc, [-1, num_filters_total_doc])
        
        with tf.name_scope('dropout'):
            self.doc_rep = tf.nn.dropout(self.h_combine_flat_doc, self.dropout)
        
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total_doc, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.doc_rep, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
            
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

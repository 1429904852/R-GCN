# encoding=utf-8

import tensorflow as tf


class CRF(object):
    def __init__(self, embedded_chars, layers, keep_prob, num_labels,
                 max_len, labels, sequence_lens, is_training):
        """
        :param embedded_chars:
        :param hidden_sizes:
        :param keep_prob:
        :param num_labels:
        :param max_len:
        :param labels:
        :param sequence_lens:
        :param is_training:
        """
        self.layers = layers
        self.keep_prob = keep_prob
        self.embedded_chars = embedded_chars
        self.max_len = max_len
        self.num_labels = num_labels
        self.labels = labels
        self.sequence_lens = sequence_lens
        self.embedding_dims = embedded_chars.shape[-1].value
        
        self.is_training = is_training

        self.mask = tf.sequence_mask(self.sequence_lens, self.max_len)

        self.l2_loss = tf.constant(0.0)

    def dense_layer(self):
        output = tf.reshape(self.embedded_chars, [-1, self.embedding_dims])
        return output, self.embedding_dims

    def output_layer(self, output, output_size):
        """
        :param output:
        :param output_size:
        :return:
        """
        with tf.variable_scope("final_output_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.get_variable("output_b", shape=[self.num_labels], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            new_logits = tf.reshape(logits, [-1, self.max_len, self.num_labels])

            return new_logits

    def cal_loss(self, new_logits):
        """
        :param new_logits:
        :return:
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=new_logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.sequence_lens)
            return tf.reduce_mean(-log_likelihood), trans

    def get_pred(self, new_logits, trans_params=None):
        """
        :param new_logits:
        :param trans_params:
        :return:
        """
        with tf.name_scope("maskedOutput"):
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(new_logits, trans_params,
                                                                        self.sequence_lens)
            return viterbi_sequence

    def construct_graph(self):
        """
        :return:
        """
        output, output_size = self.dense_layer()
        new_logits = self.output_layer(output, output_size)
        loss, trans_params = self.cal_loss(new_logits)
        pred_y = self.get_pred(new_logits, trans_params)

        true_y = tf.boolean_mask(self.labels, self.mask, name="masked_true_y")
        pred_y = tf.boolean_mask(pred_y, self.mask, name="masked_pred_y")

        return (loss, true_y, pred_y)







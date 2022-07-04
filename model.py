import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from bert import modeling
from bert import optimization
from crf import CRF
from module import multi_head_attention, GCN

class BertNer(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_classes = config["num_classes"]
        self.__learning_rate = config["learning_rate"]
        self.__ner_layers = config["ner_layers"]
        self.__k = config["top_k"]
        self.picture_length = config["picture_length"]
        self.picture_dimension = config["picture_dimension"]
        self.__max_len = config["sequence_length"]
        self.multi_heads = config["multi_heads"] 
        self.__random_base = config["random_base"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step
        
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="label_ids")
        self.sequence_len = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_len")
        self.picture_id = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='picture_id')
        self.pt_simi_score_id = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='pt_simi_score_id')
        self.pt_image_id = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name='pt_image_id')
        self.pp_simi_score_id = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='pp_simi_score_id')
        self.pp_image_id = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name='pp_image_id')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")
        
        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)
        output_layer = model.get_sequence_output()
        embedding_dims = output_layer.shape[-1].value
        output_picture_feature = self.picture_id
        inputs_pic = tf.reshape(output_picture_feature, [-1, self.picture_length, self.picture_dimension])
        pt_image_embedding = self.pt_image_id
        inputs_pt_image = tf.reshape(pt_image_embedding, [-1, self.picture_length * self.__k, self.picture_dimension])
        pp_image_embedding = self.pp_image_id
        inputs_pp_image = tf.reshape(pp_image_embedding, [-1, self.picture_length * self.__k, self.picture_dimension])

        pt_simi_score_matrix = self.pt_simi_score_id
        input_pt_simi = tf.reshape(pt_simi_score_matrix, [-1, self.__k, self.__k])
        pp_simi_score_matrix = self.pp_simi_score_id
        input_pp_simi = tf.reshape(pp_simi_score_matrix, [-1, self.__k, self.__k])
        
        inputs_pt_image = tf.reshape(inputs_pt_image, [-1, self.__k * self.picture_length, self.picture_dimension])        
        output_pt = GCN(inputs_pt_image, self.__k, input_pt_simi, self.picture_length, self.picture_dimension, self.__random_base, "pt")
        output_pt_sen_img = tf.squeeze(tf.reduce_mean(output_pt, 1))
        output_pt_sen_img = tf.reshape(output_pt_sen_img, [-1, self.picture_length, self.picture_dimension])
        output_pt_sen_img = tf.reshape(output_pt_sen_img, [-1, self.picture_dimension // 2])
        weight_image_to_text = tf.get_variable(name="image_to_text",shape=[self.picture_dimension // 2, embedding_dims],
        initializer=tf.random_uniform_initializer(-self.__random_base, self.__random_base))
        output_pt_sen_img = tf.matmul(output_pt_sen_img, weight_image_to_text)
        
        output_pt_sen_img = tf.reshape(output_pt_sen_img, [-1, embedding_dims])
        output_layer = tf.reshape(output_layer, [-1, embedding_dims])
        gate_weight_pt_image = tf.get_variable(name="gate_weight_pt_image", shape=[embedding_dims, embedding_dims], initializer=tf.random_uniform_initializer(-self.__random_base, self.__random_base))
        gate_pt = tf.sigmoid(tf.matmul(output_layer, gate_weight_pt_image) + tf.matmul(output_pt_sen_img, gate_weight_pt_image))
        gate_pt = tf.reshape(gate_pt, [-1, self.__max_len, embedding_dims])
        output_pt_sen_img = tf.reshape(output_pt_sen_img, [-1, self.__max_len, embedding_dims])
        output_layer = tf.reshape(output_layer, [-1, self.__max_len, embedding_dims])
        output_pt_sen_img = output_layer + gate_pt * output_pt_sen_img
        
        inputs_pp_image = tf.reshape(inputs_pp_image, [-1, self.__k * self.picture_length, self.picture_dimension])
        output_pp = GCN(inputs_pp_image, self.__k, input_pp_simi, self.picture_length, self.picture_dimension, self.__random_base, "pp")
        output_pp_img_img = tf.squeeze(tf.reduce_mean(output_pp, 1))

        output_pp_img_img = tf.reshape(output_pp_img_img, [-1, self.picture_dimension])
        inputs_pic = tf.reshape(inputs_pic, [-1, self.picture_dimension])
        gate_weight_pp_image = tf.get_variable(name="gate_weight_pp_image", shape=[self.picture_dimension, self.picture_dimension], initializer=tf.random_uniform_initializer(-self.__random_base, self.__random_base))
        gate_pp = tf.sigmoid(tf.matmul(inputs_pic, gate_weight_pp_image) + tf.matmul(output_pp_img_img, gate_weight_pp_image))
        gate_pp = tf.reshape(gate_pp, [-1, self.picture_length, self.picture_dimension])
        output_pp_img_img = tf.reshape(output_pp_img_img, [-1, self.picture_length, self.picture_dimension])
        inputs_pic = tf.reshape(inputs_pic, [-1, self.picture_length, self.picture_dimension])
        output_pp_img_img = inputs_pic + gate_pp * output_pp_img_img
        
        if self.__is_training:
            output_pt_sen_img = tf.nn.dropout(output_pt_sen_img, keep_prob=0.9)
        output_layer_sen = multi_head_attention(output_pt_sen_img, output_pt_sen_img, output_pt_sen_img, embedding_dims, self.multi_heads, scope="self_multihead_sen")
        output_layer_imag = multi_head_attention(output_pp_img_img, output_pp_img_img, output_pp_img_img, self.picture_dimension, self.multi_heads, scope="self_multihead_imag")
        output_layer_sen = multi_head_attention(output_layer_imag, output_layer_imag, output_layer_sen, embedding_dims, self.multi_heads, scope="cross_multihead")         
        
        ner_model = CRF(embedded_chars=output_layer_sen,
                              layers=self.__ner_layers,
                              keep_prob=self.keep_prob,
                              num_labels=self.__num_classes,
                              max_len=self.__max_len,
                              labels=self.label_ids,
                              sequence_lens=self.sequence_len,
                              is_training=self.__is_training)

        self.loss, self.true_y, self.predictions = ner_model.construct_graph()
        
        if self.__is_training:
            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_rate):
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.picture_id: batch["picture_id"],
                     self.pt_simi_score_id: batch["pt_simi_score_id"],
                     self.pt_image_id: batch["pt_image_id"],
                     self.pp_simi_score_id: batch["pp_simi_score_id"],
                     self.pp_image_id: batch["pp_image_id"],
                     self.keep_prob: dropout_rate}
        _, loss, true_y, predictions = sess.run([self.train_op, self.loss, self.true_y, self.predictions],
                                                feed_dict=feed_dict)
        return loss, true_y, predictions

    def eval(self, sess, batch):
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.picture_id: batch["picture_id"],
                     self.pt_simi_score_id: batch["pt_simi_score_id"],
                     self.pt_image_id: batch["pt_image_id"],
                     self.pp_simi_score_id: batch["pp_simi_score_id"],
                     self.pp_image_id: batch["pp_image_id"],
                     self.keep_prob: 1.0}

        loss, true_y, predictions = sess.run([self.loss, self.true_y, self.predictions], feed_dict=feed_dict)
        return loss, true_y, predictions

    def infer(self, sess, batch):
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.picture_id: batch["picture_id"],
                     self.pt_simi_score_id: batch["pt_simi_score_id"],
                     self.pt_image_id: batch["pt_image_id"],
                     self.pp_simi_score_id: batch["pp_simi_score_id"],
                     self.pp_image_id: batch["pp_image_id"],
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
    
    def infer_1(self, sess, batch):
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.picture_id: batch["picture_id"],
                     self.pt_simi_score_id: batch["pt_simi_score_id"],
                     self.pt_image_id: batch["pt_image_id"],
                     self.pp_simi_score_id: batch["pp_simi_score_id"],
                     self.pp_image_id: batch["pp_image_id"],
                     self.keep_prob: 1.0}

        true_y, predict = sess.run([self.true_y, self.predictions], feed_dict=feed_dict)

        return true_y, predict

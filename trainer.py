import os
import json
import argparse
import time
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from bert import modeling
from model import BertNer
from data_helper import TrainData
from metrics import mean, gen_metrics


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")
        self.data_obj = self.load_data()
        self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids, self.t_seq_len, self.t_pic_id, self.t_pt_simi_score_id, self.t_pt_image_id, self.t_pp_simi_score_id, self.t_pp_image_id, self.lab_to_idx = \
            self.data_obj.gen_data(self.config["train_data"])

        self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_lab_ids, self.e_seq_len, self.e_pic_id, self.e_pt_simi_score_id, self.e_pt_image_id, self.e_pp_simi_score_id, self.e_pp_image_id, self.lab_to_idx = \
            self.data_obj.gen_data(self.config["eval_data"], is_training=False)

        print("train data size: {}".format(len(self.t_lab_ids)))
        print("eval data size: {}".format(len(self.e_lab_ids)))

        num_train_steps = int(
            len(self.t_lab_ids) / self.config["batch_size"] * self.config["epochs"])
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def load_data(self):
        data_obj = TrainData(self.config)
        return data_obj

    def create_model(self, num_train_step, num_warmup_step):
        model = BertNer(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
        return model

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   
        with tf.Session(config=config) as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            max_f1, max_recall, max_precision = 0, 0, 0
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_in_ids,
                                                      self.t_in_masks,
                                                      self.t_seg_ids,
                                                      self.t_lab_ids,
                                                      self.t_seq_len,
                                                      self.t_pic_id, 
                                                      self.t_pt_simi_score_id, 
                                                      self.t_pt_image_id, 
                                                      self.t_pp_simi_score_id, 
                                                      self.t_pp_image_id):

                    loss, true_y, predictions = self.model.train(sess, batch, self.config["keep_prob"])

                    f1, precision, recall = gen_metrics(true_y, predictions, self.lab_to_idx)
                    print("train: step: {}, loss: {}, recall: {}, precision: {}, f1: {}".format(
                        current_step, loss, recall, precision, f1))

                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_recalls = []
                        eval_precisions = []
                        eval_f1s = []

                        for eval_batch in self.data_obj.next_batch(self.e_in_ids,
                                                                   self.e_in_masks,
                                                                   self.e_seg_ids,
                                                                   self.e_lab_ids,
                                                                   self.e_seq_len,
                                                                   self.e_pic_id,
                                                                   self.e_pt_simi_score_id, 
                                                                   self.e_pt_image_id, 
                                                                   self.e_pp_simi_score_id, 
                                                                   self.e_pp_image_id):
                            eval_loss, eval_true_y, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                            f1, precision, recall = gen_metrics(eval_true_y, eval_predictions, self.lab_to_idx)
                            eval_recalls.append(recall)
                            eval_precisions.append(precision)
                            eval_f1s.append(f1)
                        
                        print("\n")
                        print("eval:  loss: {}, recall: {}, precision: {}, f1: {}".format(
                            mean(eval_losses), mean(eval_recalls),
                            mean(eval_precisions), mean(eval_f1s)))
                        
                        print("\n")

                        if mean(eval_f1s) > max_f1:
                            max_f1 = mean(eval_f1s)
                            max_recall = mean(eval_recalls)
                            max_precision = mean(eval_precisions)
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            path = self.model.saver.save(sess, model_save_path, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                        print("topf1 {:g}, toprecall {:g}, topprecision {:g}".format(max_f1, max_recall, max_precision))
                        print("\n")
                        
            end = time.time()
            print("topf1 {:g}, toprecall {:g}, topprecision {:g}".format(max_f1, max_recall, max_precision))
            print("total train time: ", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()

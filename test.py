import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import tensorflow as tf
from model import BertNer
from bert import tokenization
from metrics import get_chunk, gen_metrics, mean, gen_metrics_type
import io
import argparse
import numpy as np

class Test(object):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.batch_size = config["batch_test_size"]
        self.output_path = config["output_path"]
        self.vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.label_to_index = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]
        self.create_model()
        self.load_graph()

    def load_vocab(self):        
        with open(os.path.join(self.output_path, "label_to_index.json"), "r") as f:
            label_to_index = json.load(f)

        return label_to_index

    @staticmethod
    def read_data(file_path):
        text, ner_anno, image = [], [], []
        pt_simi_score, pt_image = [], []
        pp_simi_score, pp_image = [], []
        lines = io.open(file_path, "r", encoding="UTF-8").readlines()
        for i in range(0, len(lines), 7):
            text.append(lines[i].strip().lower().split())
            ner_anno.append(lines[i + 1].strip().split())
            image.append(lines[i + 2].strip())

            pt_scores = lines[i + 3].strip().split()[:5]
            pt_scores = [float(score) for score in pt_scores]
            pt_adj = np.zeros((5, 5),dtype=np.float)
            np.fill_diagonal(pt_adj, 1)
            for j in range(5):
                pt_adj[0][j] = np.array(pt_scores[j])
            for j in range(5):
                pt_adj[j][0] = np.array(pt_scores[j])
            pt_simi_score.append(pt_adj)

            ptimage = lines[i + 4].strip().split()[:5]
            ptimage = [image.split('.')[0] for image in ptimage]
            ptimage = "##".join(ptimage) + '.jpg'
            pt_image.append(ptimage)

            pp_scores = lines[i + 5].strip().split()[:5]
            pp_scores = [float(score) for score in pp_scores]
            pp_adj = np.zeros((5, 5), dtype=np.float)
            np.fill_diagonal(pp_adj, 1)
            for j in range(5):
                pp_adj[0][j] = np.array(pp_scores[j])
            for j in range(5):
                pp_adj[j][0] = np.array(pp_scores[j])
            pp_simi_score.append(pp_adj)

            ppimage = lines[i + 6].strip().split()[:5]
            ppimage = [image.split('.')[0] for image in ppimage]
            ppimage = "##".join(ppimage) + '.jpg'
            pp_image.append(ppimage)
        return text, ner_anno, image, pt_simi_score, pt_image, pp_simi_score, pp_image

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        labels_ids = [[label_to_index[item] for item in label] for label in labels]
        return labels_ids
    
    def padding(self, input_ids, input_masks, segment_ids, label_ids, label_to_index):
        pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len = [], [], [], [], []
        for input_id, input_mask, segment_id, label_id in zip(input_ids, input_masks, segment_ids, label_ids):
            if len(input_id) < self.sequence_length:
                pad_input_ids.append(input_id + [0] * (self.sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self.sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self.sequence_length - len(segment_id)))
                pad_label_ids.append(label_id + [label_to_index["O"]] * (self.sequence_length - len(label_id)))
                sequence_len.append(len(input_id))
            else:
                pad_input_ids.append(input_id[:self.sequence_length])
                pad_input_masks.append(input_mask[:self.sequence_length])
                pad_segment_ids.append(segment_id[:self.sequence_length])
                pad_label_ids.append(label_id[:self.sequence_length])
                sequence_len.append(self.sequence_length)

        return pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len
    
    def sentence_to_idx(self, inputs, labels):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        new_labels = []

        for text, label in zip(inputs, labels):

            tokens = []
            new_label = []
            for token, tag in zip(text, label):
                token = tokenizer.tokenize(token)
                tokens.extend(token)
                if len(token) == 1:
                    new_label.extend([tag])
                elif len(token) > 1:
                    new_label.extend([tag] + ["X"] * (len(token)-1))

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)

            label = ["O"] + new_label + ["O"]

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
            new_labels.append(label)
        
        labels_ids = self.trans_label_to_index(new_labels, self.label_to_index)

        input_ids, input_masks, segment_ids, labels_ids, sequence_len = self.padding(input_ids,
                                                                                      input_masks,
                                                                                      segment_ids,
                                                                                      labels_ids,
                                                                                      self.label_to_index)

        return input_ids, input_masks, segment_ids, labels_ids, sequence_len
    
    def load_graph(self):
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def picture_feature(self, pictures, __pictures_path):
        pictures_id = []
        for picture in pictures:
            photo_feature_path = __pictures_path + picture.split(".")[0] + '.npy'
            photo_features = np.load(photo_feature_path)
            pictures_id.append(photo_features)
        pictures_id = np.array(pictures_id)
        print(pictures_id.shape)
        return pictures_id

    def create_model(self):
        self.model = BertNer(config=self.config, is_training=False)

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id):
        z = list(zip(input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id))
        input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id = zip(*z)

        num_batches = len(input_ids) // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_label_ids = label_ids[start: end]
            batch_sequence_len = sequence_len[start: end]
            batch_picture_id = picture_id[start: end]
            batch_pt_simi_score_id = pt_simi_score_id[start: end]
            batch_pt_image_id = pt_image_id[start: end]
            batch_pp_simi_score_id = pp_simi_score_id[start: end]
            batch_pp_image_id = pp_image_id[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids=batch_label_ids,
                       sequence_len=batch_sequence_len,
                       picture_id=batch_picture_id,
                       pt_simi_score_id=batch_pt_simi_score_id,
                       pt_image_id=batch_pt_image_id,
                       pp_simi_score_id=batch_pp_simi_score_id,
                       pp_image_id=batch_pp_image_id)
    
    def predict(self, batch):
        true_label, prediction = self.model.infer_1(self.sess, batch)
        prediction_tag = [self.index_to_label[i] for i in list(prediction)]
        true_tag = [self.index_to_label[i] for i in list(true_label)]

        chunks = get_chunk(prediction, self.label_to_index)
        return prediction_tag, true_tag, true_label, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    with open(args.config_path, "r") as fr:
        config = json.load(fr)
    predictorer = Test(config)
    
    text, labels, image, pt_simi_score, pt_image, pp_simi_score, pp_image = predictorer.read_data(config["test_data"])
    input_ids, input_masks, segment_ids, labels_ids, sequence_len = predictorer.sentence_to_idx(text, labels)
    picture_id = predictorer.picture_feature(image, config["pictures_path"])
    
    pt_simi_score_id = pt_simi_score
    pt_image_id = predictorer.picture_feature(pt_image, config["pt_pictures_path"])
    pp_simi_score_id = pp_simi_score
    pp_image_id = predictorer.picture_feature(pp_image, config["pp_pictures_path"])

    test_recalls, test_precisions, test_f1s = [], [], []

    ORG_test_recalls, ORG_test_precisions, ORG_test_f1s = [], [], []
    PER_test_recalls, PER_test_precisions, PER_test_f1s = [], [], []
    LOC_test_recalls, LOC_test_precisions, LOC_test_f1s = [], [], []
    OTHER_test_recalls, OTHER_test_precisions, OTHER_test_f1s = [], [], []
    MISC_test_recalls, MISC_test_precisions, MISC_test_f1s = [], [], []
    
    test_predictions_total, test_true_y_total = [], []

    pre_lable = []
    true_label = []

    for test_batch in predictorer.next_batch(input_ids, input_masks, segment_ids, labels_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id):
        prediction_tag, true_tag, true_y, predictions = predictorer.predict(test_batch)
        true_tag = [iter_val for iter_val in true_tag]
        true_tag = true_tag[1:-1]
        true_label.append(" ".join(true_tag))

        prediction_tag = [iter_val for iter_val in prediction_tag]
        prediction_tag = prediction_tag[1:-1]
        pre_lable.append(" ".join(prediction_tag))

        f1, precision, recall = gen_metrics(true_y, predictions, predictorer.label_to_index)
        test_recalls.append(recall)
        test_precisions.append(precision)
        test_f1s.append(f1)

        PER_f1, PER_precision, PER_recall = gen_metrics_type(true_y, predictions, predictorer.label_to_index, 'PER')
        PER_test_recalls.append(PER_recall)
        PER_test_precisions.append(PER_precision)
        PER_test_f1s.append(PER_f1)

        LOC_f1, LOC_precision, LOC_recall = gen_metrics_type(true_y, predictions, predictorer.label_to_index, 'LOC')
        LOC_test_recalls.append(LOC_recall)
        LOC_test_precisions.append(LOC_precision)
        LOC_test_f1s.append(LOC_f1)

        ORG_f1, ORG_precision, ORG_recall = gen_metrics_type(true_y, predictions, predictorer.label_to_index, 'ORG')
        ORG_test_recalls.append(ORG_recall)
        ORG_test_precisions.append(ORG_precision)
        ORG_test_f1s.append(ORG_f1)

        if config["dataset"] == "twitter15":
            OTHER_f1, OTHER_precision, OTHER_recall = gen_metrics_type(true_y, predictions, predictorer.label_to_index, 'OTHER')
            OTHER_test_recalls.append(OTHER_recall)
            OTHER_test_precisions.append(OTHER_precision)
            OTHER_test_f1s.append(OTHER_f1)
        
        if config["dataset"] == "twitter17":
            MISC_f1, MISC_precision, MISC_recall = gen_metrics_type(true_y, predictions, predictorer.label_to_index, 'MISC')
            MISC_test_recalls.append(MISC_recall)
            MISC_test_precisions.append(MISC_precision)
            MISC_test_f1s.append(MISC_f1)

    print("Overall test:  recall: {}, precision: {}, f1: {}".format(mean(test_recalls), mean(test_precisions), mean(test_f1s)))
    print("PER test:  recall: {}, precision: {}, f1: {}".format(mean(PER_test_recalls), mean(PER_test_precisions), mean(PER_test_f1s)))
    print("LOC test:  recall: {}, precision: {}, f1: {}".format(mean(LOC_test_recalls), mean(LOC_test_precisions), mean(LOC_test_f1s)))
    print("ORG test:  recall: {}, precision: {}, f1: {}".format(mean(ORG_test_recalls), mean(ORG_test_precisions), mean(ORG_test_f1s)))
    if config["dataset"] == "twitter15":
        print("OTHER test:  recall: {}, precision: {}, f1: {}".format(mean(OTHER_test_recalls), mean(OTHER_test_precisions), mean(OTHER_test_f1s)))
    if config["dataset"] == "twitter17":
        print("MISC test:  recall: {}, precision: {}, f1: {}".format(mean(MISC_test_recalls), mean(MISC_test_precisions), mean(MISC_test_f1s)))
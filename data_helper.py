import os
import json
import random
import sys
from itertools import chain
import io
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np

from bert import tokenization

class TrainData(object):
    def __init__(self, config):
        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        self.__pictures_path = config["pictures_path"]
        self.__pt_pictures_path = config["pt_pictures_path"]
        self.__pp_pictures_path = config["pp_pictures_path"]
        
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]
        self._batch_size = config["batch_size"]

    @staticmethod
    def read_data(file_path):
        """
        :param file_path:
        :return
        """
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

    def trans_to_index(self, inputs, labels):
        """
        :param inputs
        :param labels
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
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
            # print(tokens)
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            # print(input_id)
            label_n = ["O"] + new_label + ["O"]

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
            new_labels.append(label_n)

        return input_ids, input_masks, segment_ids, new_labels

    def picture_feature(self, pictures, __pictures_path):
        pictures_id = []
        for picture in pictures:
            photo_feature_path = __pictures_path + picture.split(".")[0] + '.npy'
            photo_features = np.load(photo_feature_path)
            pictures_id.append(photo_features)
        pictures_id = np.array(pictures_id)
        print("pictures_id")
        print(pictures_id.shape)
        return pictures_id
    
    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        :param labels
        :param label_to_index
        :return:
        """
        labels_ids = [[label_to_index[item] for item in label] for label in labels]
        return labels_ids

    def padding(self, input_ids, input_masks, segment_ids, label_ids, label_to_index):
        """
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids
        :param label_to_index
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len = [], [], [], [], []
        for input_id, input_mask, segment_id, label_id in zip(input_ids, input_masks, segment_ids, label_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))

                pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))

                pad_label_ids.append(label_id + [label_to_index["O"]] * (self._sequence_length - len(label_id)))

                sequence_len.append(len(input_id))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

                pad_label_ids.append(label_id[:self._sequence_length])

                sequence_len.append(self._sequence_length)

        return pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len

    def gen_data(self, file_path, is_training=True):
        """
        :param file_path:
        :param is_training:
        :return:
        """
        inputs, labels, image, pt_simi_score, pt_image, pp_simi_score, pp_image = self.read_data(file_path)
        
        print("read finished")

        if is_training:
            uni_label = list(set(chain(*labels)))
            uni_label.append("X")
            print(uni_label)
            label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))
            with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as fw:
                json.dump(label_to_index, fw, indent=0, ensure_ascii=False)
        else:
            with open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf8") as fr:
                label_to_index = json.load(fr)

        inputs_ids, input_masks, segment_ids, labels = self.trans_to_index(inputs, labels)

        print("index transform finished")
        labels_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")
        picture_id = self.picture_feature(image, self.__pictures_path)
        pt_simi_score_id = pt_simi_score
        pt_image_id = self.picture_feature(pt_image, self.__pt_pictures_path)
        pp_simi_score_id = pp_simi_score
        pp_image_id = self.picture_feature(pp_image, self.__pp_pictures_path)
        inputs_ids, input_masks, segment_ids, labels_ids, sequence_len = self.padding(inputs_ids,
                                                                                      input_masks,
                                                                                      segment_ids,
                                                                                      labels_ids,
                                                                                      label_to_index)

        for i in range(5):
            print("input: ", inputs[i])
            print("input_id: ", inputs_ids[i])
            print("input_mask: ", input_masks[i])
            print("segment_id: ", segment_ids[i])
            print("label_id: ", labels_ids[i])

        return inputs_ids, input_masks, segment_ids, labels_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id, label_to_index

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id):
        """
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :param sequence_len:
        :param picture_id
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id))
        random.shuffle(z)
        input_ids, input_masks, segment_ids, label_ids, sequence_len, picture_id, pt_simi_score_id, pt_image_id, pp_simi_score_id, pp_image_id = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
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

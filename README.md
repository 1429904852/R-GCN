# R-GCN
Code and data for "Learning from Different text-image Pairs: A Relation-enhanced Graph Convolutional Network for Multimodal NER" (ACM MM 2022)

### Datasets

Please download the processed image features from [this site](https://pan.baidu.com/s/1QQHdX2R98F_k7OqtG3upbQ?pwd=0olr), and keep the path of the dataset consistent with the filetree. Besides, you can also download the pre-trained model from [this site](https://pan.baidu.com/s/1QQHdX2R98F_k7OqtG3upbQ?pwd=0olr) and put it into folder uncased_L-12_H-768_A-12.

```sh
├── /data/
│  ├── /twitter2015/
│  │  │  ├── /images2015_feature/
│  │  │  ├── /twitter2015_images/
│  │  │  ├── /img2img_sim_topk_vec_2015/
│  │  │  ├── /img2text_sim_topk_vec_2015/
│  │  ├── train_2.txt
│  │  ├── valid_2.txt
│  │  ├── test_2.txt
│  ├── /twitter2017/
│  │  │  ├── /images2017_feature/
│  │  │  ├── /twitter2017_images/
│  │  │  ├── /img2img_sim_topk_vec_2017/
│  │  │  ├── /img2text_sim_topk_vec_2017/
│  │  ├── train_2.txt
│  │  ├── valid_2.txt
│  │  ├── test_2.txt
```

### Dependencies

```bash
+ python 3.6
+ tensorflow 1.14.0
+ numpy 1.14.5
```


### Usage

```bash
sh run_15.sh
sh run_17.sh
```

### Citation

If the code is used in your research, please cite our paper.
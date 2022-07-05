# R-GCN
Code and data for "Learning from Different text-image Pairs: A Relation-enhanced Graph Convolutional Network for Multimodal NER" (ACM MM 2022)

### Dataset Filetree
```sh
├── /data/
│  ├── /twitter2015/
│  │  │  ├── /images2015_feature/             //the image feature
│  │  │  ├── /twitter2015_images/             //the original image
│  │  │  ├── /img2img_sim_topk_vec_2015/      //the top-K image feature for each image
│  │  │  ├── /img2text_sim_topk_vec_2015/			//the top-K image feature for each sentence
│  │  ├── train_2.txt
│  │  ├── valid_2.txt
│  │  ├── test_2.txt
│  ├── /twitter2017/
│  │  │  ├── /images2017_feature/							//the image feature
│  │  │  ├── /twitter2017_images/							//the original image
│  │  │  ├── /img2img_sim_topk_vec_2017/			//the top-K image feature for each image
│  │  │  ├── /img2text_sim_topk_vec_2017/			//the top-K image feature for each image
│  │  ├── train_2.txt
│  │  ├── valid_2.txt
│  │  ├── test_2.txt
```

### Dependencies
```bash
python 3.6
tensorflow 1.14.0
numpy 1.14.5
```

### Datasets

Download the processed dataset from [this site](https://pan.baidu.com/s/1QQHdX2R98F_k7OqtG3upbQ?pwd=0olr), and keep the path of the dataset is consistent with the filetree. Besides, you can also download the pre-trained model from [this site](https://pan.baidu.com/s/1QQHdX2R98F_k7OqtG3upbQ?pwd=0olr).


### Usage

```bash
sh run_15.sh
sh run_17.sh
```

### Citation

If the code is used in your research, please cite our paper.
# VDEdit
This is the official PyTorch implementation for the ACM MM2024 [paper](https://arxiv.org/abs/2305.08389):

***Edit As You Wish: Video Description Editing with Multi-grained Commands***

We propose a novel Video Description Editing (VDEdit) task to automatically revise an existing video description guided by flexible user requests. To facilitate the VDEdit task, we automatically construct an open-domain dataset namely VATEX-EDIT and manually collect an e-commerce benchmark dataset called EMMAD-EDIT.

![image](https://github.com/user-attachments/assets/b5e37c2a-a2c5-466d-972b-05d860c30abb)



## Datasets
The VATEX-EDIT (EN) and EMMAD-EDIT (CN) dataset will be released publicly and can be downloaded from the [dataset page](dataset/README.md).


## Requirements


```
conda create -n VDEdit python=3.6
conda activate VDEdit
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.3.3
pip install lemminflect==0.2.1
pip install inflect==5.2.0
pip install nltk==3.6.2

```






## Training
We initialize the proposed **OPA** model with pre-trained [English BART](https://huggingface.co/facebook/bart-base) / [Chinese BART](https://huggingface.co/fnlp/bart-base-chinese) when training on the VATEX-EDIT (EN) / EMMAD-EDIT (CN) dataset.


Take training on the EMMAD-EDIT dataset as an example. Please make sure you have downloaded the `training/validation/test.json` & `middle_files/*.*` and put them under `codes/data/emmad-edit/`. 

1. **(Optional) Data Processing.**
 skip this if you download the middle_files/
```
# preprocess video features to get .tsv files to speed up training
cd codes/data_process/get_videoFeat_tsv
python create_video_tsv_EMMAD-EDIT.py
```
2. **Training & Inference**
```
cd codes/models_add_vision_cn
./train.sh
./infer.sh
```
The predicted file during inference will be put under the checkpoint folder
`./checkpoints_vision_cn/`.


## Evaluation

For VDEdit evaluation, we adopt comprehensive metrics to measure three aspects of model performance, including ***caption quality***, ***caption-command consistency***, and ***caption-video
alignment***. We also provide the *Chinese* version of above metrics. The reference repositories for all metrics are as follows:

1. ***Caption Quality*** (*Fluency*)

   PPL(GPT-2): [lm_perplexity](https://github.com/EleutherAI/lm_perplexity)
   
   BLEU4 and ROUGE-L: [coco_caption](https://github.com/ruotianluo/coco-caption/)

2. ***Caption-command Consistency*** (*Controllability*)

   SARI: [iterater](https://github.com/vipulraheja/iterater)
   
   Len-Acc / Attr-Acc / Pos-Acc: proposed in this paper

3. ***Caption-video Alignment*** (*Vision Align*)

   EMScore: [emscore](https://github.com/ShiYaya/emscore)
    [Note] Run EMScore need to modify the `encode_text()` function in CLIP/clip/model.py, please refer to the original [repo]((https://github.com/ShiYaya/emscore)).

We integrate and modify the orginal metric codes to support both EN/CN evaluation. The evaluation codes will be slightly different in English and Chinese environments. If you want to evaluate on the English data `cd metrics/eval_en`, else `cd metrics/eval_cn`.

The overall evaluation results can be obtained by running the following script and the results will be saved in `eval_log` file:
 
```
cd metrics/eval_cn
bash eval_cn_overall.sh > eval_log

# you can define the predicted file in the .sh
testfile=your_self_predicted_file
# the exp_name is to label the different predicted files
exp_name=your_exp_name
```

If you also want to print the evaluation results of breakdown commands, you can run the following script to get results of 7 specific commands:

```
bash eval_cn_breakdown.sh > eval_log_bk
```


## Citation

```
@article{yao2023edit,
  title={Edit As You Wish: Video Description Editing with Multi-grained Commands},
  author={Yao, Linli and Zhang, Yuanmeng and Wang, Ziheng and Hou, Xinglin and Ge, Tiezheng and Jiang, Yuning and Jin, Qin},
  journal={arXiv preprint arXiv:2305.08389},
  year={2023}
}
```

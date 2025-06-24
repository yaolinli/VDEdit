# Dataset

VATEX-EDIT and EMMAD-EDIT are two datasets that support  **Video Description Editing** task.  Each data sample is a quadruple *(video, command, reference caption, edited caption)* . 

You can download the whole dataset annotation files and video features [here(preparing)](). All files are structured as the followings:

```
dataset/
├── EMMAD-EDIT
│   ├── clip_cn_feats/
│   ├── example_videos/
│   ├── middle_files/
│   ├── abstract_attr/
│   ├── training.json
│   ├── validation.json
│   └── test.json
└── VATEX-EDIT
    ├── blip_en_feats/
    ├── clip_en_feats/
    ├── example_videos/
    ├── middle_files/
    ├── training.json
    ├── validation.json
    └── test.json

```

## VATEX-EDIT

It is automatically built upon the English video captioning dataset [VATEX (EN)](https://arxiv.org/pdf/1904.03493.pdf) describing over 600 human activities. The VATEX-EDIT dataset contains 34,269 videos and 1,057,956 *(video, command, reference caption, edited caption)* quadruples. We follow the original VATEX dataset to get training, validation and public test split according to video ids. 

| #Videos                | #Editing instances         |
| ---------------------- | -------------------------- |
| Train / Val / Test     | Train / Val / Test         |
| 25,467 / 2,935 / 5,867 | 784,805 / 91,513 / 181,638 |

**Annotation format**:

```
{
    "vid": video id, 
    "dtype": specific type of editing command,
    "command": operation (add/delete), 
    "attr": edited attribute, 
    "atype": type of attribute (verb/noun/modifier), 
    "oldcap": reference caption, 
    "reference": positioned reference using <mask> if assigning positions, 
    "newcap": edited caption,
}
```
| dtype       | global_len_add  | global_len_dele | global_attr_add    | global_attr_dele   | local_len_add     | local_len_dele    | local_attr_add   |
| ----------- | --------------- | --------------- | ------------------ | ------------------ | ----------------- | ----------------- | ---------------- |
| **command** | *<add, - , - >* | *<del, - , - >* | *<add, - , attr >* | *<del, - , attr >* | *<add, pos , - >* | *<del, pos , - >* | *<add,pos,attr>* |

**Download**



**[Note]** Due to the legal and privacy concerns, we cannot directly share the original videos from VATEX dataset. You can get the related original videos following the instructions of [VATEX dataset website](https://eric-xw.github.io/vatex-website/download.html).

**Data Example**
<video src="./VATEX-EDIT/example_videos/QAgcSr8Khus_000012_000022.mp4"></video>
```
{
    "vid": "QAgcSr8Khus_000012_000022", 
    "dtype": "local_attr_add", 
    "command": "<add>", 
    "attr": "drop", 
    "atype": "verb", 
    "oldcap": "a man pets a cat that 's there .", 
    "reference": "a man <mask> pets a cat that 's there ."
    "newcap": ["a man drops some garbage off outdoors and pets a cat that 's there ."], 
}
```


## EMMAD-EDIT

It is manually collected based on the Chinese E-commerce video captioning dataset [E-MMAD](https://e-mmad.github.io/e-mmad.net/index.html). Given the product video, the original advertising video description and external product information, we further manually annotate video description editing samples. 

The E-MMAD dataset has overall 23,960 editing instances for 12,295 product videos with two remarkable characteristics, i.e. long videos/descriptions and diverse attributes. The average video length is 27.1 seconds and the average description length is around 100 words.  

|                   | #Videos               | #Editing instances     |
| ----------------- | --------------------- | ---------------------- |
|   EMMAD-EDIT                 | Train / Val / Test    | Train / Val / Test     |
| *specific subset*        | 16,176 / 5,418 / 5,502 | 31,610 / 10,586 / 10,737 |
| *abstract subset* | 15,955 / 5,328 / 5,432 | 15,959 / 5,328 / 5,432  |

**Annotation format:**

```
{
    "vid": video id, 
    "dtype": specific type of editing command,
    "command": operation (add/delete), 
    "attr": edited attribute, 
    "atype": type of attribute (specific/abstract), 
    "oldcap": reference caption, 
    "reference": positioned reference using <mask> if assigning positions, 
    "newcap": edited caption,
    "allattr": product structure information,
    "video_url": url to download the original video,
    "video_title": video title,
}
```
| dtype       | global_len_add  | global_len_dele | global_attr_add    | global_attr_dele   | local_len_add     | local_len_dele    | local_attr_add   |
| ----------- | --------------- | --------------- | ------------------ | ------------------ | ----------------- | ----------------- | ---------------- |
| **command** | *<add, - , - >* | *<del, - , - >* | *<add, - , attr >* | *<del, - , attr >* | *<add, pos , - >* | *<del, pos , - >* | *<add,pos,attr>* |

**Download**

We provide the frame clip features  and you can download the original video using the video url. The more challenging subset of *abstract attribute*  is put under `EMMAD-EDIT/abstract_attr/`.

**Data Example**
<video src="EMMAD-EDIT/example_videos/200563409291.mp4"></video>
```
{
    "vid": 200563409291, 
    "dtype": "local_attr_add", 
    "command": "<add>", 
    "attr": "实用,俏皮", 
    "atype": "specific", 
    "oldcap": "经典水桶包，子母包设计，再现繁盛时代的小鹿包。外形酷似水桶，包身圆润，别致的子母包设计，追求简约、时尚，凸现自我的创意个性与个人色彩。", 
    "reference": "经典水桶包，子母包设计，再现繁盛时代的小鹿包。外形酷似水桶，包身圆润<mask>，别致的子母包设计，追求简约<mask>、时尚，凸现自我的创意个性与个人色彩。", 
    "newcap": "经典水桶包，子母包设计，再现繁盛时代的小鹿包。外形酷似水桶外 型，包身圆润又不失俏皮的造型，别致的子母包设计，追求简约实用的时尚，凸现自我的创意个性与个人色彩。"
    "allattr": "品类:斜挎包,单肩包,水桶包,女包;时间季节:2020;新品:新款;风格:时尚,休闲,简约,潮流,欧美时尚;修饰:手提,小鹿;人群:女士;上市时间:2018年春夏;大小:中;箱包硬度:软;款式:单肩包;里料材质:织物;背包方式:单肩斜挎手提;内部结构:手机袋,证件袋,拉链暗袋;品牌:VANESSA HOGAN;颜色分类:香草白1,黑色,香草白,婴儿粉;皮革材质:牛皮;是否可折叠:否;适用场景:休闲;图案:纯色;质地:牛皮;流行元素:车缝线;货号:VH1804158020405;肩带样式:单根;形状:水桶形;销售渠道类型:商场同款(线上线下都销售);流行款式名称:水桶包;成色:全新;提拎部件类型:软把;闭合方式:磁扣;适用对象:青年;有无夹层:无;", 
}
```

import pdb
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import os
import os.path as op
import json
import shutil
import argparse
import nltk
from transformers import BertTokenizer
import sys
sys.path.append("./")
from utils import *
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")


parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
parser.add_argument('--inpath', type=str, default='', help='the path of the generated file.')
parser.add_argument('--exp_name', type=str, default='', help='add vision or none')
args = parser.parse_args()


inpath = args.inpath
out_root = "./results/{}/".format(args.exp_name)
out_prefile = "./predict_files/{}/".format(args.exp_name)

dtypes = ["global_len_dele", "global_len_add", "global_attr_add", "global_attr_dele", "local_len_add", "local_len_dele", "local_attr_add", "all"]
rawdatas = read_json_line(inpath)
if os.path.exists(out_prefile):
    os.system("rm -rf {}*".format(out_prefile))
else:
    os.system("mkdir {}".format(out_prefile))
if os.path.exists(out_root):
    os.system("rm -rf {}*".format(out_root))
else:
    os.system("mkdir {}".format(out_root))
datas = {}
for dt in dtypes:
    datas[dt] = []

# bread down the test file to 7 specific command files according to the "dtype"
split_num = []
all_datas = []
for jterm in tqdm(rawdatas):
    command = jterm["command"]
    reference = jterm["reference"]
    dtype = None 
    if "dtype" in jterm and jterm["dtype"] is not None:
        dtype = jterm["dtype"]
    else:
        dtype = find_dtype(jterm)

    assert (dtype in dtypes) == True
    if isinstance(jterm["newcap_gt"], list):
         jterm["newcap_gt"] = " ".join(jterm["newcap_gt"])
         
    reference = tokenizer.convert_ids_to_tokens(tokenizer.encode(reference, return_tensors='pt')[0], skip_special_tokens=True)
    reference = " ".join(reference)     
    oldcap = tokenizer.convert_ids_to_tokens(tokenizer.encode(jterm["oldcap"], return_tensors='pt')[0], skip_special_tokens=True)
    oldcap = " ".join(oldcap)    
    if "newcap_gt_case" in jterm:
        gtcaps = tokenizer.convert_ids_to_tokens(tokenizer.encode(jterm["newcap_gt_case"], return_tensors='pt')[0], skip_special_tokens=True)
    else:
        gtcaps = tokenizer.convert_ids_to_tokens(tokenizer.encode(jterm["newcap_gt"], return_tensors='pt')[0], skip_special_tokens=True)
    gtcaps = " ".join(gtcaps)  
    if "new_generated_case" in jterm:
        gencap = tokenizer.convert_ids_to_tokens(tokenizer.encode(jterm["newcap_generated_case"], return_tensors='pt')[0], skip_special_tokens=True) 
    else:
        gencap = tokenizer.convert_ids_to_tokens(tokenizer.encode(jterm["newcap_generated"], return_tensors='pt')[0], skip_special_tokens=True) 
    gencap = " ".join(gencap)  
    jterm["reference"] = reference
    jterm["oldcap"] = oldcap
    jterm["newcap_gt"] = gtcaps
    jterm["newcap_generated"] = gencap
    datas[dtype].append(jterm)
    all_datas.append(jterm)
datas["all"] = all_datas
assert len(rawdatas) == len(all_datas)

# save split files
namei = 0
for i, dtype in enumerate(dtypes):
    k =  dtype
    v = datas[k]
    print("#", k, len(v))
    if dtype != "all":
        split_num.append(len(v))
    if len(v) > 0:
        write_json_line(f"{out_prefile}{namei}_{k}.json", v)
        namei += 1

assert sum(split_num) == len(rawdatas)

# get cocoformat file 
for i, dtype in enumerate(dtypes):
    k =  dtype
    v = datas[k]
    iid = 0
    anno_data = []
    gen_data = []
    img_data  = []
    for i, jterm in enumerate(v):
        vid = "input" + str(i) + "_" + str(jterm["vid"]) # unique identity for each (video, command, oldcap)
        gtcaps = jterm["newcap_gt"]
        gencap = jterm["newcap_generated"]
        assert isinstance(gtcaps, str) == True
        gtcaps = [gtcaps]
        img_data.append({"filename":str(jterm["vid"]), "id":vid})
        gen_data.append({"image_id": vid, "caption": gencap})
        for gtcap in gtcaps:
            json_item = {}
            json_item["image_id"] = vid
            json_item["caption"] = gtcap
            json_item["id"] = iid
            iid += 1
            anno_data.append(json_item)
            
    # gt coco format
    json_data = {}
    json_data["annotations"] = anno_data
    json_data["images"] = img_data   
    json_data.update({"type": "captions", "info": "dummy", "licenses": "dummy"})
    # genereated file
    out_file = out_root + '{}.json'.format(dtype)
    out_json = out_root + '{}.caption_coco_format.json'.format(dtype)
    write_json(out_json, json_data)
    write_json(out_file, gen_data)

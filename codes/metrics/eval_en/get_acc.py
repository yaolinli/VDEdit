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
from nltk.tokenize import word_tokenize
import sys
sys.path.append("./")
from utils import *

dtype2idx = {
    "global_len_dele":0, "global_len_add":1, "global_attr_add":2, "global_attr_dele":3, "local_len_add":4, "local_len_dele":5, "local_attr_add":6, "all":7,
}

parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
parser.add_argument('--inpath', type=str, default='', help='the path of the generated file.')
parser.add_argument('--exp_name', type=str, default='', help='add vision or none')
parser.add_argument('--dtype', type=str, default='', help='add vision or none')
args = parser.parse_args()


inpath = args.inpath
out_root = "results/{}/".format(args.exp_name)
out_prefile = "./predict_files/{}/".format(args.exp_name)

if args.dtype == "overall":
    dtypes = ["all"]
else:
    dtypes = ["global_len_dele", "global_len_add", "global_attr_add", "global_attr_dele", "local_len_add", "local_len_dele", "local_attr_add", "all"]


# calculate len acc/attr acc/pos acc
files = os.listdir(out_prefile)
total_pos_correct = []
all_attr_acc = []
all_pos_acc = []
all_len_acc = []
for i, dtype in enumerate(dtypes):
    if len(dtypes) == 8 and dtype == "all":
        continue
    cfile =  str(dtype2idx[dtype]) + "_" + dtype + ".json"
    if cfile not in files:
        continue
    print("==========================================")
    print("===>", cfile)
    bad_cases = []
    good_cases = []
    length_change = []
    attr_include = {}
    pos_correct = []
    N = 1
    datas = read_json_line(out_prefile+cfile)
    for i, jterm in tqdm(enumerate(datas)):
        vid = "input" + str(i) + "_" + jterm["vid"]     # unique identity for each (video, command, oldcap)
        command = jterm["command"]
        oldcap = jterm["oldcap"]
        gencap = jterm["newcap_generated"]
        reference = jterm["reference"]
        gtcaps = jterm["newcap_gt"]
        attr = jterm["attr"]
        atype = jterm["atype"]
        dtype = jterm["dtype"]
        if atype not in attr_include:
            attr_include[atype] = []

        # calculate command acc
        clens0 = get_clen(oldcap)
        clens1 = get_clen(gencap)
        if command == "<add>" and (clens1 - clens0) >= N:
            length_change.append(1)
        elif command == "<dele>" and (clens0 - clens1) >= N:
            length_change.append(1)
        else:
            length_change.append(0)
            
        # calculate pos acc
        if ("local" in dtype and "add" in dtype):
            flag, mask_content = align_pos(reference, gencap) # 0 or 1
            pos_correct.append(flag)
            if flag:
                gencap = mask_content
            else:
                gencap = ""
        # caculate attr appear acc
        if ("attr" in dtype):
            if attr is None:
                continue
            attrs = attr.split(",")
            for attr_i in attrs:
                if command == "<add>" and is_in(attr_i, gencap):
                    attr_include[atype].append(1) 
                    good_cases.append(jterm)
                elif command == "<dele>" and not_in(attr_i, gencap):
                    attr_include[atype].append(1) 
                    good_cases.append(jterm)
                else:
                    attr_include[atype].append(0) 
                    bad_cases.append(jterm)

    print("Length command acc: {:.2f}".format(sum(length_change) / len(length_change)))
    all_len_acc.extend(length_change)
    
    if ("attr" in cfile) or ("all" in cfile):
        print("Attr appear acc: ")
        total_attr_include = []
        attr_accs = {}
        for k,v in attr_include.items():
            total_attr_include.extend(v)
            if len(v) > 0:
                attr_accs[k] = sum(v)/len(v)
            else:
                attr_accs[k] = 0.0
        attr_accs["total"] = sum(total_attr_include) / len(total_attr_include)
        # print("total {:.2f}  [verb {:.2f}/ noun {:.2f}/ modifier{:.2f}]".format(attr_accs["total"], attr_accs["verb"], attr_accs["noun"], attr_accs["modifier"]))

        print("total {:.2f} ".format(attr_accs["total"]))
        all_attr_acc.extend(total_attr_include)
        
    if ("local" in cfile and "add" in cfile) or ("all" in cfile):
        print("POS appear acc: {:.2f}".format(sum(pos_correct) / len(pos_correct)))
        all_pos_acc.extend(pos_correct)
    
if len(dtypes) == 8:
    print("==========================================")
    print("===>", "7_all.json")
    print("Len Acc: {:.2f}".format(sum(all_len_acc) / len(all_len_acc)))
    print("Attr Acc {:.2f} ".format(sum(all_attr_acc) / len(all_attr_acc)))
    print("Pos Acc: {:.2f}".format(sum(all_pos_acc) / len(all_pos_acc)))
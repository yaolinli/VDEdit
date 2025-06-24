import json
import pdb
import os
import os.path as op
import shutil
import numpy as np
from tqdm import tqdm
import re
import random
random.seed(1234)
import pdb
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")


def get_dist(w1, w2):
    dist = -1
    if w1 == w2:
        dist = 0
    elif "mask" in w1 and len(w1) > len("mask"):
        dist = 10
    else:
        dist = 100
    return dist

def filter_repetitive(align_path, ref, gen):
    history = None
    rm_idx = []
    for idx, (i,j) in enumerate(align_path):
        if idx == 0:
            history = (i,j, idx)
            continue
        if j == history[1]:
            cur_dist = get_dist(ref[i], gen[j])
            his_dist = get_dist(ref[history[0]], gen[history[1]])
            if cur_dist < his_dist:
                rm_idx.append(history[2])
                history = (i,j, idx)
            else:
                rm_idx.append(idx)
            continue
        history = (i,j, idx)
    newpath = [align_path[i] for i in range(len(align_path)) if i not in rm_idx]
    return newpath

def DTWAlign(s1, s2):
    DTW={}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    
    PATH = {}
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= get_dist(s1[i], s2[j])
            min_dist = min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            # min_dist = min(DTW[(i, j-1)], DTW[(i-1, j-1)])
            DTW[(i, j)] = dist + min_dist
            for pos in [(i-1, j), (i, j-1), (i-1, j-1)]:
            # for pos in [(i, j-1), (i-1, j-1)]:
                if DTW[pos] == min_dist:
                    PATH[(i, j)] = pos
                    break
    best_path = []
    i, j = len(s1)-1, len(s2)-1
    best_path.append((i,j))
    while True:
        i,j = PATH[(i,j)]
        best_path.append((i,j))
        if i == 0 and j == 0:
            break
    return best_path[::-1]


def align_pos(reference, gencap, attr=None):
    '''
    reference: "a man raises a hammer <mask> breaking it ."
    gencap:   "a man raises a hammer and hits the bullseye repeatedly breaking it ."
    '''
    # pdb.set_trace()
    reference = reference.replace("< mask >", "<mask>")
    refs = reference.split(" ")
    reference = []
    map_dict = {}
    mi = 0
    for i, tok in enumerate(refs):
        if tok == "<mask>":
            mask_i = "mask"+str(mi)
            reference.append(mask_i)
            map_dict[mask_i] = []
            mi += 1
        else:
            reference.append(tok)
    reference = " ".join(reference)
    # print("[ref]", reference)
    # print("[gen]", gencap)
    ref_tokens = reference.split(" ")
    gencap_tokens = gencap.split(" ")
    align_path = DTWAlign(ref_tokens, gencap_tokens)
    # print("[ref]", ref_tokens)
    # print("[gen]", gencap_tokens)
    
    align_path = filter_repetitive(align_path, ref_tokens, gencap_tokens)
    for (i,j) in align_path:
        if ref_tokens[i] in map_dict:
            map_dict[ref_tokens[i]].append(gencap_tokens[j])
    # print(map_dict)
    
    flag = 1
    for k,v in map_dict.items():
        if len(v) == 0:
            flag = 0
            return flag, None
        
    mask_content = " ".join(" ".join(v) for k,v in map_dict.items())
    # print(mask_content)
    if attr is not None and len(attr) > 0:
        # print("attr:", attr)
        attrs = attr.split(",")
        for attr in attrs:
            if attr not in mask_content:
                flag = 0
                return flag, None
    return flag, mask_content


def is_in(attr_sent, sent):
    attrs = attr_sent.split(",")
    in_num = 0
    sent = sent.replace(" ", "")
    for attr in attrs:
        if attr in sent:
            in_num += 1
    return in_num == len(attrs) 

def not_in(attr_sent, sent):
    attrs = attr_sent.split(",")
    sent = sent.replace(" ", "")
    in_num = 0
    for attr in attrs:
        if attr in sent:
            in_num += 1
    return in_num == 0

def find_dtype(jterm):
    dtype = ["global", "len", "add"] # or "local"
    attr = jterm["attr"]
    reference = jterm["reference"]
    command = jterm["command"]
    if attr is not None:
        dtype[1] = "attr"
        if ("<mask>" in reference):
            dtype[0] = "local"
            dtype[2] = "add"
        else:
            dtype[0] = "global"
            if command == "<dele>":
                dtype[2] = "dele"
    else:
        dtype[1] = "len"
        if command == "<dele>":
            dtype[2] = "dele"
            if reference != jterm["oldcap"]:
                dtype[0] = "local"
        else:
            dtype[2] = "add"
            if "<mask>" in reference:
                dtype[0] = "local"
    return " ".join(dtype)
  
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    return data

def read_json_line(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data

def write_json(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

def write_json_line_format(path, data):
    with open(path, "w", encoding="utf-8") as fout:
        for line in data:
            fout.writelines(json.dumps(line, indent=4, ensure_ascii=False)+"\n")
    return 

def write_json_line(path, data):
    with open(path, "w", encoding="utf-8") as fout:
        for line in data:
            fout.writelines(json.dumps(line, ensure_ascii=False)+"\n")
    return 

def dele_multispace(sent):
    for i in range(10, 1, -1):
        space = "".join([" "]*i)
        sent = sent.replace(space, " ") 
    return sent

def get_clen(cap):
    return len(cap.split(" "))

def addstop(rawstr):
    if not rawstr.endswith("."):
        return rawstr + " ."
    else:
        return rawstr

def get_minEditDis_sent(sent, gts):
    # get a sentence with minimum edit distance in the given list gts
    min_dist = 100000.0
    min_sent = ""
    for gt in gts:
        dist = calculate_edit_distance(sent, gt)/len(gt.split(" "))
        if dist < min_dist:
            min_dist = dist
            min_sent = gt
    return min_sent
    

if __name__ == "__main__":
    reference = "此款皮鞋精选<mask>头层牛皮，采用擦色<mask>打磨工艺；鞋帮选用松紧套脚，穿着上更加的便捷。内里采用的是猪皮内里鞋垫，贴合脚型，保证了鞋垫的透气性。此款切尔西靴仅有黑色设计，黑色系能更好的展现腿部与脚部的线条感。"
    gencap = "此款高帮皮鞋精选优质头层牛皮，采用擦色手工打磨工艺；鞋帮选用的是松紧套脚，穿着上更加的便捷。内里采用的是猪皮内里鞋垫，贴合脚型，保证了鞋垫的透气性。此款切尔西靴仅有黑色设计，黑色系能更好的展现腿部与脚部的线条感。"
    reference = tokenizer.convert_ids_to_tokens(tokenizer.encode(reference, return_tensors='pt')[0], skip_special_tokens=True)
    reference = " ".join(reference)     
    gencap = tokenizer.convert_ids_to_tokens(tokenizer.encode(gencap, return_tensors='pt')[0], skip_special_tokens=True)
    gencap = " ".join(gencap) 
    attr = None
    flag, mask_content = align_pos(reference, gencap, attr=attr)
    print("==> reference:")
    print(reference)
    print("==> gencap:")
    print(gencap)
    print("==> pos align & aligned content")
    print(flag, mask_content)
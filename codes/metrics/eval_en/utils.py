import json
import pdb
import os
import os.path as op
import shutil
import numpy as np
from tqdm import tqdm
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
import spacy
nlp = spacy.load("en_core_web_sm")
import random
random.seed(1234)
import pdb

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
    ref_tokens = word_tokenize(reference)
    gencap_tokens = word_tokenize(gencap)
    align_path = DTWAlign(ref_tokens, gencap_tokens)

    
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
    doc = nlp(sent)
    tokens = [tok.text for tok in doc]
    proto_tokens = [tok.lemma_ for tok in doc]
    attrs = attr_sent.split(",")
    in_num = 0
    for attr in attrs:
        if attr in sent:
            in_num += 1
        elif attr in proto_tokens:
            in_num += 1
    return in_num == len(attrs) 

def not_in(attr_sent, sent):
    doc = nlp(sent)
    tokens = [tok.text for tok in doc]
    proto_tokens = [tok.lemma_ for tok in doc]
    attrs = attr_sent.split(",")
    in_num = 0
    for attr in attrs:
        if attr in sent:
            in_num += 1
        elif attr in proto_tokens:
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
        f.write(json.dumps(data, indent=4))

def write_json_line_format(path, data):
    with open(path, "w", encoding="utf-8") as fout:
        for line in data:
            fout.writelines(json.dumps(line, indent=4)+"\n")
    return 


def write_json_line(path, data):
    with open(path, "w", encoding="utf-8") as fout:
        for line in data:
            fout.writelines(json.dumps(line)+"\n")
    return 


def random_span(delecap, attr):
    doc = nlp(delecap)
    tokens = [tok.text for tok in doc]
    proto_tokens = [tok.lemma_ for tok in doc]
    if attr in proto_tokens:
        idx = proto_tokens.index(attr)
        start = random.randint(0, idx+1)
        end = random.randint(idx+1, len(tokens))
        return " ".join(tokens[start: end])
    else:
        return attr

def mask_span(delList, attrList, sent):
    for delecap in delList:
        dele = delecap
        for attr in attrList:
            if attr in delecap:
                dele = random_span(delecap, attr)
                break
        sent = sent.replace(dele, "")
    return sent
    
def replace_mask(sent):
    # modify <mask N> -> <mask>
    # e.g. "A boy launches two rockets across the gym <mask1> ."
    #      --> "A boy launches two rockets across the gym <mask> ."
    pattern = re.compile(r'<mask\d>')
    masks = pattern.findall(sent)
    # pdb.set_trace()
    # print(masks)
    for mask in masks:
        sent = sent.replace(mask, '<mask>')
    # merge continuous <mask>
    # e.g. "a man holding a <mask> <mask> ball rolling it <mask> on his hands and arms ."
    #      "a man holding a <mask> ball rolling it <mask> on his hands and arms ."
    for i in range(10, 1, -1):
        mask = " ".join(["<mask>"]*i)
        sent = sent.replace(mask, "<mask>")
    return sent

def dele_multispace(sent):
    for i in range(10, 1, -1):
        space = "".join([" "]*i)
        sent = sent.replace(space, " ") 
    return sent

def get_clen(cap):
    tokens = word_tokenize(cap)
    return len(tokens)

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
    
    
def format_data(raw_data, command=None, split=None, attr=False): # type=<add> or <dele> or "None"
    print("format data length ...")
    data = []
    for term in tqdm(raw_data):
        vid = term["vid"]
        oldcap = term["oldcap"]
        newcap = term["newcap"]
        '''
        output format example:
        {
            "vid": "G9zN5TTuGO4_000179_000189", 
            "delecap": "with the help of a rope", 
            "attr": "help", 
            "atype": "modifier", 
            "command": "<add>", 
            "pos": "40", 
            "reference": "A person is climbing down a snowy cliff <mask>.", 
            "oldcap": "A person is climbing down a snowy cliff .", 
            "newcap": "A person is climbing down a snowy cliff with the help of a rope.",
            "attr_ids": [8, 9, 10, 11, 12, 13]
        }
        '''

        jterm = {}
        jterm["vid"] = vid
        jterm["command"] = None
        jterm["reference"] = None
        jterm["oldcap"] = None
        jterm["newcap"] = None
        jterm["attr"] = None
        jterm["atype"] = None
        jterm["delecap"] = None
        jterm["pos"] = None
        jterm["attr_ids"] = None

        if command in ["<add>", "<dele>"]:
            jterm["oldcap"] = oldcap
            jterm["newcap"] = newcap
            jterm["command"] = command
            jterm["reference"] = oldcap
            if command == "<add>":
                if isinstance(newcap, str):
                    assert get_clen(oldcap) < get_clen(newcap)
                if isinstance(newcap, list):
                    assert get_clen(oldcap) < get_clen(random.sample(newcap, 1)[0])
            else:
                if isinstance(newcap, str):
                    assert get_clen(oldcap) > get_clen(newcap)
                if isinstance(newcap, list):
                    assert get_clen(oldcap) > get_clen(random.sample(newcap, 1)[0])
        else:
            # default: oldcap < newcap
            shortcap = oldcap
            longcap = newcap
            assert get_clen(shortcap) < get_clen(longcap)
            # randomly half <add> half <dele>
            if random.random() < 0.5: 
                jterm["oldcap"] = shortcap
                jterm["newcap"] = longcap
                jterm["command"] = "<add>"
                jterm["reference"] = shortcap
            else:
                jterm["oldcap"] = longcap
                jterm["newcap"] = shortcap 
                jterm["command"] = "<dele>"
                jterm["reference"] = longcap
        data.append(jterm)
    # check "." at the end
    for i, term in enumerate(data):
        data[i]["oldcap"] = addstop(term["oldcap"])
        data[i]["reference"] = addstop(term["reference"])
        if isinstance(term["newcap"], str):
            data[i]["newcap"] = addstop(term["newcap"])
        elif isinstance(term["newcap"], list):
            newcap = [addstop(sent) for sent in term["newcap"]]
            data[i]["newcap"] = newcap
    return data


def format_data_global(raw_data, split=None): # type=<add> or <dele> or "None"
    print("format data global ...")
    data = []
    for term in tqdm(raw_data):
        vid = term["vid"]
        oldcap = term["oldcap"]
        newcap = term["newcap"]
        '''
        output format example:
        {
            "vid": "G9zN5TTuGO4_000179_000189", 
            "delecap": "with the help of a rope", 
            "attr": "help", 
            "atype": "modifier", 
            "command": "<add>", 
            "pos": "40", 
            "reference": "A person is climbing down a snowy cliff <mask>.", 
            "oldcap": "A person is climbing down a snowy cliff .", 
            "newcap": "A person is climbing down a snowy cliff with the help of a rope.",
            "attr_ids": [8, 9, 10, 11, 12, 13]
        }
        '''

        jterm = {}
        jterm["vid"] = vid
        jterm["command"] = term["command"]
        jterm["reference"] = term["reference"]
        jterm["oldcap"] = term["oldcap"]
        jterm["attr"] = term["attr"]
        jterm["atype"] = term["atype"]
        assert jterm["attr"] is not None
        assert jterm["atype"] is not None
        jterm["delecap"] = term["delecap"]
        jterm["pos"] = None
        jterm["attr_ids"] = None
        jterm["newcap"] = None
        # update newcap: construct -> extend gts
        gts = term["gts"]
        if split == "test":
            jterm["newcap"] = gts
        else: # training/validation set
            # rselect a gt(min edit distance) from gts for training
            jterm["newcap"] = get_minEditDis_sent(jterm["oldcap"], gts)
            # get pos 
            jterm["pos"] = None #TODO
            # get attr_ids
            jterm["attr_ids"] = None #TODO
        
        data.append(jterm)
    # check "." at the end
    for i, term in enumerate(data):
        data[i]["oldcap"] = addstop(term["oldcap"])
        data[i]["reference"] = addstop(term["reference"])
        if isinstance(term["newcap"], str):
            data[i]["newcap"] = addstop(term["newcap"])
        elif isinstance(term["newcap"], list):
            newcap = [addstop(sent) for sent in term["newcap"]]
            data[i]["newcap"] = newcap
    return data


def format_data_local(raw_data, command=None, attr=False, split=None): 
    print("format data local ...")
    data = []
    for term in tqdm(raw_data):
        vid = term["vid"]
        oldcap = term["oldcap"]
        newcap = term["newcap"]
        '''
        output format example:
        {
            "vid": "G9zN5TTuGO4_000179_000189", 
            "delecap": "with the help of a rope", 
            "attr": "help", 
            "atype": "modifier", 
            "command": "<add>", 
            "pos": "40", 
            "reference": "A person is climbing down a snowy cliff <mask>.", 
            "oldcap": "A person is climbing down a snowy cliff .", 
            "newcap": "A person is climbing down a snowy cliff with the help of a rope.",
            "attr_ids": [8, 9, 10, 11, 12, 13]
        }
        '''
        jterm = {}
        jterm["vid"] = vid
        jterm["attr"] = None
        jterm["atype"] = None
        # correct add&dele command
        if command != term["command"]:
            jterm["command"] = command
            jterm["oldcap"] = term["newcap"]
            jterm["newcap"] = term["oldcap"]
        else:
            jterm["command"] = command
            jterm["oldcap"] = term["oldcap"]
            jterm["newcap"] = term["newcap"]
        if command == "<dele>":
            assert get_clen(jterm["oldcap"]) > get_clen(jterm["newcap"])
        elif command == "<add>":
            assert get_clen(jterm["oldcap"]) < get_clen(jterm["newcap"])
        
        if attr:
            jterm["attr"] = term["attr"]
            jterm["atype"] = term["atype"]
            assert jterm["attr"] is not None
            assert jterm["atype"] is not None
        jterm["delecap"] = term["delecap"]
        jterm["pos"] = None
        jterm["attr_ids"] = None
        jterm["reference"] = None
        # indicate pos in the reference
        if command == "<add>":
            ref = replace_mask(term["rawdata"]["reference"])
            ref = dele_multispace(ref)
            jterm["reference"] = ref
        elif command == "<dele>":
            jterm["reference"] = mask_span(term["rawdata"]["delList"], term["rawdata"]["mainToken"], jterm["oldcap"])
            # print(term["rawdata"]["mainToken"])
            # print(term["rawdata"]["delList"])
            # print(jterm["oldcap"])
            # print(jterm["reference"])
            # print(jterm["newcap"])
            # pdb.set_trace()
        # test set: newcap -> list
        if split == "test":
            jterm["newcap"] = [jterm["newcap"]]
        
        data.append(jterm)
    # check "." at the end and lower()
    for i, term in enumerate(data):
        data[i]["oldcap"] = addstop(term["oldcap"]).lower()
        data[i]["reference"] = addstop(term["reference"]).lower()
        if isinstance(term["newcap"], str):
            data[i]["newcap"] = addstop(term["newcap"]).lower()
        elif isinstance(term["newcap"], list):
            newcap = [addstop(sent).lower() for sent in term["newcap"]]
            data[i]["newcap"] = newcap
    return data


if __name__ == "__main__":
    ref = "a woman gives a demonstration <mask> to come to her <mask> sessions ."
    cap = "a woman is giving a demonstration and invitation to come to her gong therapy sessions ."
    acc, _ = align_pos(ref, cap)
    print(acc)
    # acc, _ = align_pos(ref, cap, "hammer")
    # print(acc)
    # acc, _ = align_pos(ref, cap, "hammer,hand")
    # print(acc)
    # acc, _ = align_pos(ref, cap, "hammer,repeatedly")
    # print(acc)
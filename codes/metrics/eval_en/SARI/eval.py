from sari import *
import argparse
import csv
import json
import numpy as np
import spacy
from tqdm import tqdm
import pdb 

_KWARGS_DESCRIPTION = """
Calculates sari score (between 0 and 100) given a list of source and predicted
sentences, and a list of lists of reference sentences.
Args:
    sources: list of source sentences where each sentence should be a string.
    predictions: list of predicted sentences where each sentence should be a string.
    references: list of lists of reference sentences where each sentence should be a string.
Returns:
    sari: sari score
Examples:
    >>> sources=["About 95 species are currently accepted ."]
    >>> predictions=["About 95 you now get in ."]
    >>> references=[["About 95 species are currently known .","About 95 species are now accepted .","95 species are now accepted ."]]
    >>> sari = datasets.load_metric("sari")
    >>> results = sari.compute(sources=sources, predictions=predictions, references=references)
    >>> print(results)
    {'sari': 26.953601953601954}
"""
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

if __name__ == '__main__':

    # SARI
    # sources=["A man is showing the products he uses shine shoes ."]
    # predictions=["A man is showing the products he uses to  shine shoes."]
    # # predictions=["A man is showing the products he uses to clean and shine shoes."]
    # references=[["a guy is explaining the items you need to clean a pair of shoes .",
    #              "a man is showing the products he uses to clean and shine shoes ."]]

    parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
    parser.add_argument('--inpath', type=str, default='', help='the path of the generated file.')
    args = parser.parse_args()

    inpath = args.inpath
    datas = read_json_line(inpath)
    
    sari_scores, keep_scores, delete_scores, add_scores = [], [], [], []
    for jterm in tqdm(datas):
        reference = jterm["reference"]
        for sep_tok in ["<mask>", "< mask >", "[MASK]", "[ MASK ]"]:
            reference = reference.replace(sep_tok, "")
        sources = [reference]
        gts = jterm["newcap_gt"]
        if isinstance(gts[0], list):
            references = jterm["newcap_gt"]
        else:
            references = [jterm["newcap_gt"]]
        predictions = [jterm["newcap_generated"]]

        sari, keep, delete, add = compute(sources=sources, predictions=predictions, references=references)
        sari_scores.append(sari)
        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)
    
    sari_score = sum(sari_scores) / len(sari_scores)
    keep = sum(keep_scores) / len(keep_scores)
    add = sum(add_scores) / len(add_scores)
    delete = sum(delete_scores) / len(delete_scores)
    print("SARI: {:.1f}, KEEP: {:.3f}, ADD: {:.3f}, DELETE: {:.3f}".format(sari_score, keep, add, delete))
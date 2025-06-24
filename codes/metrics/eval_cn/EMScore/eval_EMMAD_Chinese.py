import os
import argparse
import pickle
import numpy as np
import json
import glob
import torch
import math
from tqdm import tqdm
from emscore_Chinese import EMScorer
from emscore_Chinese.utils import get_idf_dict, compute_correlation_uniquehuman
import pdb

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        
def get_feats_dict(feat_dir_path, video_ids):
    print('loding cache feats ........')
    file_path_list = glob.glob(feat_dir_path+'/*.npy')
    feats_dict = {}
    for file_path in tqdm(file_path_list):
        vid = file_path.split('/')[-1][:-4]
        if vid in video_ids:
            data = np.load(file_path)
            feats_dict[vid] = torch.tensor(data)
    return feats_dict

def read_json_line(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', default='../../../dataset/EMMAD-EDIT/', type=str, help='The path you storage VATEX-EVAL feats')
    parser.add_argument('--vid_base_path', default='../../../dataset/EMMAD-EDIT/videos/', type=str, help='The path you storage VATEX-EVAL videos (optinal, if you use prepared video feats, You do not need to consider this)')
    parser.add_argument('--use_n_refs', default=1, type=int, help='How many references do you want to use for evaluation (1~9)')
    parser.add_argument('--use_feat_cache', default=True, action='store_true', help='Whether to use pre-prepared video features')
    parser.add_argument('--use_idf', action='store_true', default=False)
    parser.add_argument('--inpath', default='../coco_caption/predict_files_mix/local_len_add.json', type=str)
    opt = parser.parse_args()

    """
    Dataset prepare
    """
    video_ids = None
    vid_base_path = opt.vid_base_path  # optional
    if opt.inpath == "":
        samples_list = pickle.load(open(os.path.join(opt.storage_path, 'candidates_list.pkl'), 'rb'))
        gts_list = pickle.load(open(os.path.join(opt.storage_path, 'gts_list.pkl'), 'rb'))
        video_ids = pickle.load(open(os.path.join(opt.storage_path, 'video_ids.pkl'), 'rb')) 
        cands = samples_list.tolist() # 18000  generated captions; list [cand1, cand2, ... ]
        refs = gts_list.tolist() # 18000 gt captions; list [[ref1, .., ref10], [...], ...]
    else:
        datas = read_json_line(opt.inpath)
        cands = []
        refs = []
        video_ids = [] 
        for jterm in datas:
            cands.append(jterm["newcap_generated"].replace(" ", ""))
            refs.append([jterm["newcap_gt"].replace(" ", "")])
            video_ids.append(str(jterm["vid"]))

    """
    Video feats prepare
    """
    use_uniform_sample = -1
    if not opt.use_feat_cache:
        vids = [vid_base_path+vid+'.mp4' for vid in video_ids]
        metric = EMScorer(vid_feat_cache=[])
    else:
        if 'clip_cn_feats' in opt.storage_path:
            vid_clip_feats_dir = opt.storage_path
        else:
            vid_clip_feats_dir = os.path.join(opt.storage_path, 'clip_cn_feats')
        video_clip_feats_dict = get_feats_dict(vid_clip_feats_dir, video_ids)
        if use_uniform_sample > 0:
            for vid in video_clip_feats_dict:
                data = video_clip_feats_dict[vid]
                select_index = np.linspace(0, len(data)-1, use_uniform_sample)
                select_index = [int(index) for index in select_index]
                video_clip_feats_dict[vid] = data[select_index]
                # pdb.set_trace()

        vids = video_ids
        metric = EMScorer(vid_feat_cache=video_clip_feats_dict)
    

    """
    Prepare IDF
    """
    if opt.use_idf:
        print("calculate tf-idf ...")
        train_file = "EMMAD_train_data.txt"
        train_corpus_path = os.path.join(opt.storage_path, train_file)
        caps = []
        with open(train_corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                terms = line.strip().split("\t")
                cap = terms[-1]
                caps.append(cap)
                
        vatex_train_corpus_list = caps
        emscore_idf_dict = get_idf_dict(vatex_train_corpus_list, tokenizer, nthreads=4, language='cn')
        # max token_id are eos token id
        # set idf of eos token are mean idf value
        emscore_idf_dict[max(list(emscore_idf_dict.keys()))] = sum(list(emscore_idf_dict.values()))/len(list(emscore_idf_dict.values()))
    else:
        emscore_idf_dict = False
    

    """
    Metric calculate
    """

    results = metric.score(cands, refs=[], vids=vids, idf=emscore_idf_dict) 
    if 'EMScore(X,V)' in results:
        print('EMScore(X,V) correlation --------------------------------------')
        vid_full_res_F = results['EMScore(X,V)']['full_F']
        # vid_full_res_F: torch.Size([31844])  save the F score for each generated sent
        print('EMScore(X,V) -> full_F: {:.1f}'.format(vid_full_res_F.mean().item()*100))
        

    # if 'EMScore(X,X*)' in results:
    #     print('EMScore(X,X*) correlation --------------------------------------')
    #     refs_full_res_F = results['EMScore(X,X*)']['full_F']
    #     print('EMScore(X,X*) -> full_F: ', refs_full_res_F.mean())


    # if 'EMScore(X,V,X*)' in results:
    #     print('EMScore(X,V,X*) correlation --------------------------------------')
    #     vid_refs_full_res_F = results['EMScore(X,V,X*)']['full_F']
    #     print('EMScore(X,V,X*) -> full_F: ', vid_refs_full_res_F.mean())
        

# (Optional) extract video embeddings using Chinese CLIP
your_video_path='../../../dataset/EMMAD-EDIT/videos/'
your_storage_path='../../../dataset/EMMAD-EDIT/clip_cn_feats/'
python extract_Ch_video_embeddings.py --videos_path $your_video_path  --save_path $your_storage_path --backbone 'ViT-B/16' 

# eval Chinese EMMAD dataset using cache feats
your_storage_path='../../../dataset/EMMAD-EDIT/clip_cn_feats/'
your_eval_file='../predict_files/example/7_all.json'
python eval_EMMAD_Chinese.py --storage_path $your_storage_path --use_feat_cache  \
--inpath $your_eval_file

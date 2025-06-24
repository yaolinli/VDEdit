# (Optional) extract video embeddings using English CLIP
your_video_path='../../../dataset/VATEX-EDIT/videos/'
your_storage_path='../../../dataset/VATEX-EDIT/clip_en_feats/'
python extract_En_video_embeddings.py --videos_path $your_video_path  --save_path $your_storage_path --backbone 'ViT-B/16' 

# eval English VATEX dataset using cache feats
your_storage_path='../../../dataset/VATEX-EDIT/clip_en_feats/'
your_eval_file='../predict_files/example/7_all.json'
python eval_VATEX_English.py --storage_path $your_storage_path --use_feat_cache  \
--inpath $your_eval_file

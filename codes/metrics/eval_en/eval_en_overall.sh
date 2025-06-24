# conda env: base
# set the file path to eval
testfile="test_example.json"
exp_name="example" # e.g. "opa"
clip_en_feats="/home/yaolinli/dataset/VATEX-EDIT/clip_en_feats/"

file_name="7_all"
file="all"
result_root="../results/$exp_name/"  
predict_root="../predict_files/$exp_name/"

# split & format test file
python get_format.py --inpath=$testfile --exp_name=$exp_name

# get Length Acc/Attr Acc/POS Acc
python get_acc.py --exp_name=$exp_name --dtype "overall"

# get BLEU & ROUGE-L  
cd coco_caption
python cocoEvalCapDemo.py \
--testfile $result_root$file".json" \
--gtfile $result_root$file".caption_coco_format.json"

# get SARI score
cd ../SARI
python eval.py --inpath $predict_root$file_name".json"

# get PPL score
cd ../lm_perplexity
python lm_perplexity/save_lm_perplexity_data.py \
    --model_config_path preset_configs/gpt2_medium.json \
    --data_path $predict_root$file_name".json"

# get EMScore
cd ../EMScore
python eval_VATEX_English.py --use_feat_cache \
    --inpath $predict_root$file_name".json"  --storage_path $clip_en_feats

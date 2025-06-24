function eval(){  
    for file_name in ` ls $1 `  
    do  
        predict_root="../"$1
        result_root="../"$2
        file_name=${file_name: 0: 0-5} # "7_all"
        file=${file_name: 2} # "all"
        clip_en_feats=$4
        echo $file_name

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
        CUDA_VISIBLE_DEVICES=$3 python lm_perplexity/save_lm_perplexity_data.py \
            --model_config_path preset_configs/gpt2_medium.json \
            --data_path $predict_root$file_name".json"

        # get EMScore
        cd ../EMScore
        CUDA_VISIBLE_DEVICES=$3 python eval_VATEX_English.py --use_feat_cache --inpath $predict_root$file_name".json" --storage_path $clip_en_feats


        cd ..
    done  
}  

# conda env: base
# set the file path to eval
testfile="/data1/yll/ControlVideoCap/open-source/codes/metrics/eval_en/eval_files/vatex-edit_8-frame_test_gpt-3.5-turbo_video-info.json"
exp_name="vatex-edit_8-frame_test_gpt-3.5-turbo_video-info" # e.g. "opa"
gpu_id=1
result_root="results/$exp_name/"  
predict_root="predict_files/$exp_name/"
clip_en_feats="/data1/yll/ControlVideoCap/metrics/EMScore/VATEX-EVAL/en_clip_feats/clip_vid_feats/"

# split & format test file
python get_format.py --inpath=$testfile --exp_name=$exp_name

# get Length Acc/Attr Acc/POS Acc
python get_acc.py --exp_name=$exp_name 

# get other metrics
eval $predict_root $result_root $gpu_id $clip_en_feats
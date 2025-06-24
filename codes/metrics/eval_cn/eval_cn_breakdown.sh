function eval(){  
    for file_name in ` ls $1 `  
    do  
        predict_root="../"$1
        result_root="../"$2
        clip_cn_feats=$4
        file_name=${file_name: 0: 0-5} # "7_all"
        file=${file_name: 2} # "all"
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
            --model_config_path preset_configs/gpt2_chinese.json \
            --data_path $predict_root$file_name".json"

        # get EMScore
        cd ../EMScore
        CUDA_VISIBLE_DEVICES=$3 python eval_EMMAD_Chinese.py            --use_feat_cache \
        --inpath $predict_root$file_name".json" --storage_path $clip_cn_feats

        cd ..
    done  
}  

# conda env: base
testfile="test_example.json"
exp_name="example" # e.g. "opa"
gpu_id=0
result_root="results/$exp_name/"  
predict_root="predict_files/$exp_name/"
clip_cn_feats="/home/yaolinli/dataset/EMMAD-EDIT/clip_cn_feats/"

# split & format test file
python get_format.py --inpath=$testfile --exp_name=$exp_name

# get Length Acc/Attr Acc/POS Acc
python get_acc.py --exp_name=$exp_name 

# get other metrics
eval $predict_root $result_root $gpu_id $clip_cn_feats
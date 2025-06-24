# --model_path  your_checkpoint_path
# --val_num random_select_data_to_infer
# --num_beams 5

# Inference
python infer.py --test_mode 1 --batch_size 20 --decoding_strategy 1 --gpu 1 --use_command 1  \
      --use_attr 1  --initialization bart-base --dataset emmad-edit \
      --num_beams 5 \
      --model_path ../checkpoints_vision_cn/emmad-edit_bart-base_OPA/lr_1e-05_add_space_add_attr_use_command_use_ref \
      --exp_name _OPA  \
      --pos_type 'aligned'



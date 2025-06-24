# --model_path  your_checkpoint_path
# --val_num random_select_data_to_infer
# --num_beams 5

# beam search decoding
python infer.py --test_mode 1 --batch_size 160 --decoding_strategy 1 --gpu 1 --use_command 1  \
      --use_attr 1  --initialization bart-base --dataset vatex-edit \
      --num_beams 5 \
      --model_path ../checkpoints_vision_en/vatex-edit_bart-base_OPA/lr_1e-05_add_space_add_attr_use_command_use_aligned_pos_use_ref \
      --exp_name _OPA


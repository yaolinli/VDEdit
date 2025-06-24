# Training
python train.py --gpu 3 --batch_size 15         --test_batch_size 15 \
    --use_attr 1 \
    --dataset emmad-edit \
    --vfeat_dim 512 \
    --train 1  --epochs 12 --exp_name _OPA \
    --pos_type 'aligned'





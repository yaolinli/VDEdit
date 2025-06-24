python train.py --gpu 0 --batch_size 40 --test_batch_size 80 \
    --use_attr 1 \
    --dataset vatex-edit \
    --train 1  --epochs 12 --exp_name _OPA \
    --val_num 10000 --num_workers 4 \
    --pos_type 'aligned'


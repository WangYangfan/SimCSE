CUDA_VISIBLE_DEVICES=3 python train.py  \
    --train_dataset ../princeton_data/nli   \
    --batch_size 64 \
    --lr 3e-5   \
    --num_shards 100    \
    --save_info sup

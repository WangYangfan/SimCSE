CUDA_VISIBLE_DEVICES=1 python train.py  \
    --batch_size 64 \
    --lr 3e-5   \
    --num_shards 1000   \
    --save_info unsup

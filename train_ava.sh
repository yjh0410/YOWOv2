# Train YOWOv2 on AVA dataset
python train.py \
        --cuda \
        -d ava_v2.2 \
        -v yowo_v2_nano \
        --root /mnt/share/sda1/dataset/STAD/ \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
        --max_epoch 9 \
        --lr_epoch 3 4 5 6 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 8 \
        -accu 16 \
        -K 32

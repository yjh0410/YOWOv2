# Train YOWOv2 on UCF24 dataset
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_large \
        --root /mnt/share/ssd2/dataset/STAD/ \
        --num_workers 4 \
        --eval_epoch 1 \
        --max_epoch 7 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 8 \
        -accu 16 \
        -K 16
        # --eval \

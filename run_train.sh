LOG_PATH="./logs/train_upertnet_small.log"
rm -rf $LOG_PATH

echo "[Training start time]" `date +%T` > $LOG_PATH
echo "--------------------------------------------------" >> $LOG_PATH

CUDA_VISIBLE_DEVICES=2 python trainer.py \
    --base_model "openmmlab/upernet-convnext-small" \
    --train_batch_size 32 \
    --weight_decay 1e-4 \
    --optimizer "AdamW"\
    --scheduler_name "MultiStepLR"\
    --epochs 6 \
    --lr 2e-4 1>> $LOG_PATH 2>&1 &

echo "--------------------------------------------------" >> $LOG_PATH
echo "[Training end time]" `date +%T` >> $LOG_PATH
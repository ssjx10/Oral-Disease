
## Train

```bash
python train.py --model 'trans_unet' --vit_name $VIT_NAME --root_path $DATA_DIR --valid_path $VALID_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --angle 15
```

## Test 

```bash
python test.py --A2C_path [A2C_path] --A4C_path [A4C_path] --model 'trans_unet' --ckpt_path_A2C './transunet_model_out/test15/epoch_249.pth' --ckpt_path_A4C './transunet_model_out/test15/epoch_249.pth' --img_size 512 --batch_size 2

A2C_path와 A4C_path 수정!!


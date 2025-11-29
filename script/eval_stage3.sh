ckpt_path=out/stage3_finetuning/best_checkpoint.pth

deepspeed  --include localhost:0 fine_tuning.py \
   --batch-size 8 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset WLASL \
   --task ISLR \
   --eval 


# single gpu inference
# RGB-pose setting
# deepspeed --num_gpus 1 --master_port 29511 fine_tuning.py \
#     --batch-size 8 \
#     --gradient-accumulation-steps 1 \
#     --epochs 20 \
#     --opt AdamW \
#     --lr 3e-4 \
#     --output_dir out/test \
#     --finetune $ckpt_path \
#     --dataset WLASL \
#     --task ISLR \
#     --eval \
#     --rgb_support
# # pose-only setting
#deepspeed --num_gpus 1 --include localhost:0 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir out/test \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT \
#   --eval \
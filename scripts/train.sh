python train.py --model_name VMamba-S --model vmambav2_small_224\
    --ckpt vssm_small_0229_ckpt_epoch_222.pth --cfg  vmambav2_small_224.yaml\
    --dataset AWA2 --gamma 0.98 --input_size 448 --batch_size 32\
    --backbone_lr 1e-3 --head_lr 1e-3 --head2_lr 1e-4 --loss_L1 0.0

python train.py --model_name VMamba-S --model vmambav2_small_224\
    --ckpt vssm_small_0229_ckpt_epoch_222.pth --cfg  vmambav2_small_224.yaml\
    --dataset CUB --gamma 0.3 --input_size 448 --batch_size 32\
    --backbone_lr 1e-3 --head_lr 1e-3 --head2_lr 1e-4 --loss_L1 1.0

python train.py --model_name VMamba-S --model vmambav2_small_224\
    --ckpt vssm_small_0229_ckpt_epoch_222.pth --cfg  vmambav2_small_224.yaml\
    --dataset SUN --gamma 0.35 --input_size 448 --batch_size 32\
    --backbone_lr 1e-3 --head_lr 1e-3 --head2_lr 1e-4 --loss_L1 0.2
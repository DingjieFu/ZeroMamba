import os
import argparse


def parse_args():
    projectPath = os.path.dirname(os.path.abspath(__file__))
    datarootPath = projectPath + "/data"
    parser = argparse.ArgumentParser(description="Vision Mamba for ZSL")
    # ======================================== Path Config ======================================== #
    parser.add_argument('--pretrainedModelsRoot', type=str, 
                        default="/home/fudingjie/ZeroMamba/pretrainedModels", 
                        help='Dir for pretrained weights')
    parser.add_argument('--cfgRoot', type=str, 
                        default = projectPath + "/VisionMambaModels/VMamba/configs/vssm", 
                        help='Dir for VMamba configs(VMamba only)')
    parser.add_argument('--image_root', default= datarootPath + '/data/dataset',
                        help='Path to image root')
    parser.add_argument('--mat_path', default= datarootPath + '/data/dataset/xlsa17/data',
                        help='Features extracted from pre-training Resnet')
    parser.add_argument('--attr_path', default= datarootPath + '/data/attribute',
                        help='attribute path')
    parser.add_argument('--w2v_path', default= datarootPath + '/data/w2v', help='w2v path')
    parser.add_argument('--log_root_path', default = projectPath + '/out', help='Save path of exps')
    # ======================================== Model Config ======================================== #
    parser.add_argument('--model_name', default='VMamba-S', type=str, help='Name of model')
    parser.add_argument('--model', 
                        default='vmambav2_small_224', 
                        type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--ckpt', default = 'xxx.pth')
    """
        - Vim only
    """
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--num_classes', default=1000, type=int, help='number classes')
    """
        - Vmamba only
    """
    parser.add_argument('--cfg', type=str, metavar="FILE", 
                        default="vmambav2_small_224.yaml", help='path to config file')
    # ======================================== Other Config ======================================== #
    parser.add_argument('--dataset', default="AWA2", help='[AWA2,CUB,SUN]')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--input_size', default=448, type=int, help='image size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_classes', default=1000, type=int, help='number classes')
    parser.add_argument('--gamma', default=0.95, type=float, help='calibration ratio')
    parser.add_argument('--backbone_lr', type=float, default=1e-3, help='backbone learning rate')
    parser.add_argument('--head_lr', type=float, default=1e-3, help='classifier head learning rate')
    parser.add_argument('--head2_lr', type=float, default=1e-4, help='classifier head2 learning rate')
    parser.add_argument('--loss_L1', type=float, default=1.0, help='loss L1')
    parser.add_argument('--pre_ratio', type=float, default=1.0, help='backbone out pre ratio')
    parser.add_argument('--nepoch', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    parser.add_argument('--norm_feat_pre', default=True, help='norm region feat')
    args = parser.parse_args()
    return args

import os
import torch
import argparse
import torch.nn as nn
from VisionMambaModels.VMamba.models import build_vssm_model
from VisionMambaModels.VMamba.config import get_config


def select_model(args, use_pretrain=False):      
    if args.model_name.lower().startswith("vmamba"):
        args.cfg = os.path.join(args.cfgRoot, args.cfg)
        config = get_config(args)
        model = build_vssm_model(config)
        if use_pretrain:
            ckptPath = os.path.join(args.pretrainedModelsRoot, args.ckpt)
            checkpoint = torch.load(ckptPath, map_location="cpu")
            checkpoint_model = checkpoint['model']
            model.load_state_dict(checkpoint_model, strict=False)
    else:
        raise RuntimeError(f'Model {args.model_name} not implemented.')
    return model

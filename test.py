import os
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.helper_func import *
from parameter import parse_args
from utils.myLoader import MyDataloader, getDataset
from VisionMambaModels.model_select import select_model


args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

# ======================================== region ======================================== #
class Classifier_custom(nn.Module):
    def __init__(self, w2v_att):
        super(Classifier_custom, self).__init__() 
        use_w2v = True
        if use_w2v:
            self.init_w2v_att = F.normalize(w2v_att.clone().detach())
            self.V = nn.Parameter(self.init_w2v_att.clone(), requires_grad=True)
        else:
            self.V = nn.Parameter(nn.init.normal_(torch.empty(w2v_att.size(0), w2v_att.size(1))), requires_grad=True)

        if args.model_name.lower().endswith("b"): 
            self.dim_f = 1024
        elif args.model_name.lower().endswith("s"): 
            self.dim_f = 768
        elif args.model_name.lower().endswith("t"): 
            self.dim_f = 768
        else:
            raise ValueError("Unknown dum_f!")
        self.dim_v = 300 # w2v (x, 300)
        self.W1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)

    def forward(self, img):
        vv = F.normalize(self.V, dim=1)
        shape = img.shape
        img = img.reshape(shape[0], shape[1], shape[2]*shape[3])
        av = F.normalize(img, dim=1)
        A = torch.einsum('iv,vf,bfr->bir', vv, self.W1, av)
        A = nn.Softmax(dim=-1)(A)
        F_p = torch.einsum('bir,bfr->bif', A, av)
        Pred_att = torch.einsum('iv,vf,bif->bi', vv, self.W2, F_p)
        return Pred_att


def get_pred_feats(args, model, data_loader):
    model.eval()
    pred_feats = []
    for _, (batch_data, _) in enumerate(data_loader):
        batch_data = batch_data.to(args.device)
        with torch.no_grad():
            backbone_out_pre, backbone_out = model.VisionMamba(batch_data)
            pred_feat = model.clf_head(backbone_out)
            pred_feat_pre = model.clf_head2(model.VisionMamba.classifier2(backbone_out_pre))
            if args.norm_feat_pre:
                pred_feat_pre = l2_norm2(pred_feat_pre)
            pred_feat = pred_feat + pred_feat_pre
        pred_feats.append(pred_feat.cpu().data.numpy())
    pred_feats = np.concatenate(pred_feats, 0)
    return pred_feats

# ======================================== data prepare ======================================== #
myloader = MyDataloader(args)
# Testing Transformations
testTransform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# seen test data loader
test_seen_data = getDataset(myloader.test_seen_files, myloader.test_seen_labels, transform=testTransform)
test_seen_data_loader = DataLoader(test_seen_data, batch_size=64, shuffle=False)
# unseen test data loader
test_unseen_data = getDataset(myloader.test_unseen_files, myloader.test_unseen_labels, transform=testTransform)
test_unseen_data_loader = DataLoader(test_unseen_data, batch_size=64, shuffle=False)

use_w2v = True
if use_w2v:
    w2v_att = myloader.w2v_att
    if args.dataset == "AWA2":
        assert w2v_att.shape == (85,300)
    elif args.dataset == "CUB":
        assert w2v_att.shape == (312,300)
    elif args.dataset == "SUN":
        assert w2v_att.shape == (102,300)
    else:
        raise ValueError("Unkonwn Dataset!")
    print('Use w2v_att')

# ======================================== model config ======================================== #
VisionMamba = select_model(args, use_pretrain=False)
VisionMamba.classifier2 = Classifier_custom(w2v_att)
clf_head = nn.Linear(args.num_classes, myloader.att.size(1), bias=True)
clf_head2 = nn.Sequential(
   nn.Linear(myloader.att.size(1), 512, bias=False),
   nn.ReLU(),
   nn.Linear(512, myloader.att.size(1), bias=False)
)

model = nn.ModuleDict({
    "VisionMamba": VisionMamba,
    "clf_head": clf_head,
    "clf_head2": clf_head2
})
for param in model.parameters():
    param.requires_grad = False

model = model.to(args.device)

weightsRoot = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints/"
checkpoint_path_CZSL = weightsRoot + f"{args.dataset}_bestCZSL.pth"
model.load_state_dict(torch.load(checkpoint_path_CZSL), strict=False)
# ======================================== model config ======================================== #
seen_att = myloader.att[myloader.seenclasses]
unseen_att = myloader.att[myloader.unseenclasses]

# -------------------- Test -------------------- #
with torch.no_grad():
    unseen_pred_feats = get_pred_feats(args, model, test_unseen_data_loader)

# CZSL
zsl_unseen_sim = unseen_pred_feats @ unseen_att.T.cpu().numpy()
pred_labels = np.argmax(zsl_unseen_sim, axis=1)
zsl_unseen_pred_labels = myloader.unseenclasses[pred_labels]
CZSL = compute_accuracy(zsl_unseen_pred_labels.numpy(), 
                        myloader.test_unseen_labels.numpy(), myloader.unseenclasses.numpy())


# GZSL
checkpoint_path_GZSL = weightsRoot + f"{args.dataset}_bestGZSL.pth"
model.load_state_dict(torch.load(checkpoint_path_GZSL), strict=False)
with torch.no_grad():
    seen_pred_feats = get_pred_feats(args, model, test_seen_data_loader)
    unseen_pred_feats = get_pred_feats(args, model, test_unseen_data_loader)


# Calibrated stacking
Cs_mat = np.zeros(myloader.att.shape[0])
Cs_mat[myloader.seenclasses] = args.gamma

# GZSL
# seen classes
gzsl_seen_sim = softmax(seen_pred_feats @ myloader.att.T.cpu().numpy(), axis=1) - Cs_mat
gzsl_seen_pred_labels = np.argmax(gzsl_seen_sim, axis=1)
S = compute_accuracy(gzsl_seen_pred_labels,
                        myloader.test_seen_labels.numpy(), myloader.seenclasses.numpy())

# unseen classes
gzsl_unseen_sim = softmax(unseen_pred_feats @ myloader.att.T.cpu().numpy(), axis=1) - Cs_mat
gzsl_unseen_pred_labels = np.argmax(gzsl_unseen_sim, axis=1)
U = compute_accuracy(gzsl_unseen_pred_labels, 
                        myloader.test_unseen_labels.numpy(), myloader.unseenclasses.numpy())
H = 2*S*U / (S+U)

print('GZSL Seen=%.2f Unseen=%.2f H=%.2f || CZSL Acc=%.2f' %(S*100, U*100, H*100, CZSL*100))

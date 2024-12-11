import os
import time
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from torch import optim
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


# ---------- run log ----------
os.makedirs(args.log_root_path, exist_ok=True)
outlogDir = "{}/{}".format(args.log_root_path, args.dataset)
os.makedirs(outlogDir, exist_ok=True)
num_exps = len([f.path for f in os.scandir(outlogDir) if f.is_dir()])
outlogPath = os.path.join(outlogDir, create_unique_folder_name(outlogDir + f"/exp{num_exps}"))
os.makedirs(outlogPath, exist_ok=True)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = outlogPath + "/" + t + '.txt'
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')
logger = logging.getLogger(__name__)
argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)

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
        else:
            raise ValueError("Unknown dum_f!")
        self.dim_v = 300 # w2v (x, 300)
        self.W1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)

    def forward(self, img):
        vv = F.normalize(self.V, dim=1)
        shape = img.shape
        img = img.reshape(shape[0], shape[1], shape[2]*shape[3]) # [bs, 1024, 49]
        av = F.normalize(img, dim=1)
        # -------------------- compute Dense Attention --------------------#
        A = torch.einsum('iv,vf,bfr->bir', vv, self.W1, av)
        A = nn.Softmax(dim=-1)(A)
        F_p = torch.einsum('bir,bfr->bif', A, av)
        Pred_att = torch.einsum('iv,vf,bif->bi', vv, self.W2, F_p)
        return Pred_att


# ======================================== data prepare ======================================== #
myloader = MyDataloader(args)
# Training Transformations
trainTransform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# Testing Transformations
testTransform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# trainval data loader
trainval_data = getDataset(myloader.trainval_files, myloader.trainval_labels_new, transform=trainTransform)
trainval_data_loader = DataLoader(trainval_data, batch_size = args.batch_size)
# seen test data loader
test_seen_data = getDataset(myloader.test_seen_files, myloader.test_seen_labels, transform=testTransform)
test_seen_data_loader = DataLoader(test_seen_data, batch_size=128, shuffle=False)
# unseen test data loader
test_unseen_data = getDataset(myloader.test_unseen_files, myloader.test_unseen_labels, transform=testTransform)
test_unseen_data_loader = DataLoader(test_unseen_data, batch_size=128, shuffle=False)

use_w2v = True
if use_w2v:
    w2v_att = myloader.w2v_att
    if args.dataset == "AWA2":
        assert w2v_att.shape == (85,300)
    elif args.dataset == "CUB":
        assert w2v_att.shape == (312, 300)
    elif args.dataset == "SUN":
        assert w2v_att.shape == (102,300)
    else:
        raise ValueError("Unkonwn Dataset!")
    print('Use w2v_att')

# ======================================== model config ======================================== #
VisionMamba = select_model(args, use_pretrain=True)
VisionMamba.classifier2 = Classifier_custom(w2v_att)
clf_head = nn.Linear(args.num_classes, myloader.att.size(1), bias=False)
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
    param.requires_grad = True
print(model)

model = model.to(args.device)
logger.info(f'model: {model}')
optimizer = torch.optim.SGD([
   {"params": model.VisionMamba.parameters(), "lr": args.backbone_lr, "momentum": 0.9, "weight_decay": 0.001},
   {"params": model.clf_head.parameters(), "lr": args.head_lr, "momentum": 0.9, "weight_decay": 0.001},
   {"params": model.clf_head2.parameters(), "lr": 1e-4, "momentum": 0.9, "weight_decay": 0.001}
])

logger.info(f'optimizer: {optimizer}')
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
# ======================================== model config ======================================== #
seen_att = myloader.att[myloader.seenclasses]
unseen_att = myloader.att[myloader.unseenclasses]

best_epoch_record = [0, 0] # [CZSL, GZSL]
best_performance_record = [0.0, 0.0, 0.0, 0.0] # [U, S, H, CZSL]

for epoch in range(1, args.nepoch + 1):
    print('=' * 50)
    print('Epoch: {}'.format(epoch))
    logger.info(f'Epoch: {epoch}')

    torch.cuda.empty_cache()
    loss_meter = AverageMeter()
    model.train()
    # ----------  train  ----------
    tk = tqdm(trainval_data_loader, total=int(len(trainval_data_loader)))
    for _, (batch_data, batch_label) in enumerate(tk):
        batch_data = batch_data.to(args.device, non_blocking=True)
        batch_label = batch_label.to(args.device, non_blocking=True)
        optimizer.zero_grad()
        backbone_out_pre, backbone_out = model.VisionMamba(batch_data)
        pred_feat = model.clf_head(backbone_out)
        pred_feat_pre = model.clf_head2(VisionMamba.classifier2(backbone_out_pre))
        pred_feat = pred_feat + l2_norm2(pred_feat_pre)
        batch_att = seen_att[batch_label]
        batch_logit = pred_feat @ seen_att.T
        loss = F.cross_entropy(batch_logit, batch_label)+F.l1_loss(pred_feat, batch_att, reduction='mean')*args.loss_L1
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), batch_label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
    print('Train: Average loss: {:.4f}'.format(loss_meter.avg))
    logger.info('Train: Average loss: {:.4f}'.format(loss_meter.avg))
    lr_scheduler.step()
    # -------------------- Test -------------------- #
    def get_pred_feats(args, model, data_loader):
        model.eval()
        pred_feats = []
        for _, (batch_data, _) in enumerate(data_loader):
            batch_data = batch_data.to(args.device)
            with torch.no_grad():
                backbone_out_pre, backbone_out = model.VisionMamba(batch_data)
                pred_feat = model.clf_head(backbone_out)
                pred_feat_pre = model.clf_head2(VisionMamba.classifier2(backbone_out_pre))
                pred_feat = pred_feat + l2_norm2(pred_feat_pre)
            pred_feats.append(pred_feat.cpu().data.numpy())
        pred_feats = np.concatenate(pred_feats, 0)
        return pred_feats

    with torch.no_grad():
        seen_pred_feats = get_pred_feats(args, model, test_seen_data_loader)
        unseen_pred_feats = get_pred_feats(args, model, test_unseen_data_loader)

    # ZSL
    zsl_unseen_sim = unseen_pred_feats @ unseen_att.T.cpu().numpy()
    pred_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_pred_labels = myloader.unseenclasses[pred_labels]
    CZSL = compute_accuracy(zsl_unseen_pred_labels.numpy(), 
                            myloader.test_unseen_labels.numpy(), myloader.unseenclasses.numpy())

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
    logger.info('GZSL Seen=%.2f Unseen=%.2f H=%.2f || CZSL Acc=%.2f' %(S*100, U*100, H*100, CZSL*100))
    
    if best_performance_record[3] <= CZSL:
        best_epoch_record[0] = epoch
        best_performance_record[3] = CZSL
        torch.save(model.state_dict(), outlogPath + f"/{args.dataset}_bestCZSL.pth")
        print('BestCZSL model saved!')

    if best_performance_record[2] < H:
        best_epoch_record[1] = epoch    
        best_performance_record[0] = U
        best_performance_record[1] = S
        best_performance_record[2] = H
        torch.save(model.state_dict(), outlogPath + f"/{args.dataset}_bestGZSL.pth")
        print('BestGZSL model saved!')

    if epoch % 10 == 0:
        print('GZSL: epoch=%d, best_U=%.2f, best_S=%.2f, best_H=%.2f || CZSL: epoch=%d, best_CZSL = %.2f' % (best_epoch_record[1], best_performance_record[0]*100, best_performance_record[1]*100, best_performance_record[2]*100, best_epoch_record[0], best_performance_record[3]*100))
        logger.info('GZSL: epoch=%d, best_U=%.2f, best_S=%.2f, best_H=%.2f || CZSL: epoch=%d, best_CZSL = %.2f' % (best_epoch_record[1], best_performance_record[0]*100, best_performance_record[1]*100, best_performance_record[2]*100, best_epoch_record[0], best_performance_record[3]*100))

print('Dataset:', args.dataset, '\nGamma:',args.dataset)
print('the best GZSL epoch', best_epoch_record[1]) 
print('the best GZSL unseen accuracy is %.2f' % (best_performance_record[0]*100))
print('the best GZSL seen accuracy is %.2f' % (best_performance_record[1]*100))
print('the best GZSL H is %.2f' % (best_performance_record[2]*100))
print('the best CZSL epoch', best_epoch_record[0])
print('the best ZSL unseen accuracy is %.2f' % (best_performance_record[3]*100))

logger.info("========== END ==========")
logger.info(f'Dataset: {args.dataset}; Gamma: {args.dataset}')
logger.info(f'the best GZSL epoch: {best_epoch_record[1]}')
logger.info('the best GZSL unseen accuracy is %.2f' % (best_performance_record[0]*100))
logger.info('the best GZSL seen accuracy is %.2f' % (best_performance_record[1]*100))
logger.info('the best GZSL H is %.2f' % (best_performance_record[2]*100))
logger.info(f'the best CZSL epoch: {best_epoch_record[0]}')
logger.info('the best ZSL unseen accuracy is %.2f' % (best_performance_record[3]*100))
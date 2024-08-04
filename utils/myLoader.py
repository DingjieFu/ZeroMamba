"""
    - load data from dataset
"""
import torch
import pickle
import numpy as np
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset


class MyDataloader():
    def __init__(self, args):
        """
            - args: need
                mat_path -> xlsa17 root path
                dataset  -> dataset_name ([AWA,CUB,SUN,FLO])
                image_root -> local dataset root path
        """
        # get labels & image_names
        res101 = sio.loadmat(args.mat_path + f"/{args.dataset}/res101.mat")
        self.labels = res101['labels'].astype(np.int64).squeeze() - 1
        self.image_files = np.squeeze(res101['image_files'])
        # attribute semantic vector
        with open(args.w2v_path + f'/{args.dataset}_attribute.pkl', 'rb') as f:
            w2v_att = pickle.load(f)
        self.w2v_att = torch.from_numpy(w2v_att).float().to(args.device)

        def convert_path(image_files, img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx][0]
                if args.dataset == "AWA2":
                    image_file = img_dir + '/'.join(image_file.split('/')[5:])
                elif args.dataset == "CUB":
                    image_file = img_dir + '/'.join(image_file.split('/')[6:])
                elif args.dataset == "SUN":
                    image_file = img_dir + '/'.join(image_file.split('/')[7:])
                else:
                    raise ValueError("Unkonwn Dataset!")
                new_image_files.append(image_file)
            return np.array(new_image_files)
        # change xlsa17 image_path to current path
        self.image_files = convert_path(self.image_files, args.image_root + f"/{args.dataset}/")
        
        # get sample idxs in mat
        att_splits = sio.loadmat(args.mat_path + f"/{args.dataset}/att_splits.mat")

        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1

        self.original_att = torch.from_numpy(att_splits['original_att'].T).float().to(args.device)
        self.att = torch.from_numpy(att_splits['att'].T).float().to(args.device)
        
        # data files
        self.trainval_files = self.image_files[self.trainval_loc]
        self.test_seen_files = self.image_files[self.test_seen_loc]
        self.test_unseen_files = self.image_files[self.test_unseen_loc]

        # get label idxs in mat
        self.trainval_labels = self.labels[self.trainval_loc]
        _, self.trainval_labels_new, self.counts_trainval_labels = np.unique(
            self.trainval_labels, return_inverse=True, return_counts=True)
        self.test_seen_labels = torch.from_numpy(self.labels[self.test_seen_loc]).long()
        self.test_unseen_labels = torch.from_numpy(self.labels[self.test_unseen_loc]).long()
        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_labels.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_labels.numpy()))


class getDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_pil = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)

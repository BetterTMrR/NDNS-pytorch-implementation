import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from collections import Counter
from data_list import ImageList_idx, image_train, image_strong


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


class AvgLoss:
    def __init__(self):
        self.loss = 0.0
        self.n = 1e-8

    def add_loss(self, loss):
        self.loss += loss
        self.n += 1

    def get_avg_loss(self):
        avg_loss = self.loss / self.n
        self.n = 1e-8
        self.loss = 0.0
        return avg_loss


def cal_acc(loader, G, F1):
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader)), desc='Test'):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = F1(G(inputs))
            pred = outputs.argmax(1)
            total_num += len(labels)
            correct_num += (labels.float() == pred.cpu().float()).sum().numpy()
    return 100 * correct_num / total_num


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def adjacency_matrix(features, M1, M2):
    features = features / ((features ** 2).sum(1) ** 0.5).unsqueeze(1)
    global_idx = torch.LongTensor([i for i in range(features.size(0))])
    cosine_simi = features.mm(features.t())
    _, _topk_idx = cosine_simi.topk(dim=1, k=len(features))
    topk_idx = _topk_idx[:, 1:M1]
    connected = (topk_idx[:, :M2][topk_idx] == global_idx.unsqueeze(-1).unsqueeze(-1)).sum(2)
    edges = torch.where(connected > 0, torch.ones(connected.size()), torch.zeros(connected.size()))
    return topk_idx.detach(), edges.detach()

def memory_bank(loader, G, F1, metaidx=2, normalized=True, return_label=False):
    feature_bank = torch.zeros(len(loader.dataset), 256)
    target_bank = torch.zeros(len(loader.dataset), F1.fc.out_features)
    if return_label:
        label_bank = torch.zeros(len(loader.dataset), )
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader)), desc='Memory Bank'):
            data = iter_test.next()
            inputs = data[0].cuda()
            fea = G(inputs)
            out = F1(fea)
            feature_bank[data[metaidx]] = fea.cpu().detach()
            target_bank[data[metaidx]] = out.cpu().detach()
            if return_label:
                label_bank[data[metaidx]] = data[metaidx-1].cpu().detach().float()
    if normalized:
        feature_bank = normalize(feature_bank)
    else:
        feature_bank = feature_bank
    target_bank = F.softmax(target_bank, dim=1)
    if return_label:
        return feature_bank, target_bank, label_bank
    else:
        return feature_bank, target_bank


def get_st_feature(loaders, G, F1, args, before=''):
    src_feature_bank, src_target_bank, src_label_bank = memory_bank(loaders['src_tr'], G, F1, normalized=False, return_label=True)
    tar_feature_bank, tar_target_bank, tar_label_bank = memory_bank(loaders['tar_un'], G, F1, metaidx=3, normalized=False, return_label=True)
    np.save(args.output_dir_tgt + '/{}{}({})_src_feat_{}.npy'.format(before, args.al_strategy, args.method, args.shot), src_feature_bank)
    np.save(args.output_dir_tgt + '/{}{}({})_src_target_{}.npy'.format(before, args.al_strategy, args.method, args.shot), src_target_bank)
    np.save(args.output_dir_tgt + '/{}{}({})_src_label_{}.npy'.format(before, args.al_strategy, args.method, args.shot), src_label_bank)

    np.save(args.output_dir_tgt + '/{}{}({})_tar_feat_{}.npy'.format(before, args.al_strategy, args.method, args.shot), tar_feature_bank)
    np.save(args.output_dir_tgt + '/{}{}({})_tar_target_{}.npy'.format(before, args.al_strategy, args.method, args.shot), tar_target_bank)
    np.save(args.output_dir_tgt + '/{}{}({})_tar_label_{}.npy'.format(before, args.al_strategy, args.method, args.shot), tar_label_bank)
    print('UN: {}'.format(len(tar_target_bank)))
    if args.lbd:
        tar_feature_bank_lbd, tar_target_bank_lbd, tar_label_bank_lbd = memory_bank(loaders['tar_lbd'], G, F1, metaidx=2, return_label=True)
        print('UN: {}'.format(len(tar_target_bank_lbd)))
        np.save(args.output_dir_tgt + '/{}({})_tar_feat_lbd_{}.npy'.format(args.al_strategy, args.method, args.shot), tar_feature_bank_lbd)
        np.save(args.output_dir_tgt + '/{}({})_tar_target_lbd_{}.npy'.format(args.al_strategy, args.method, args.shot), tar_target_bank_lbd)
        np.save(args.output_dir_tgt + '/{}({})_tar_label_lbd_{}.npy'.format(args.al_strategy, args.method, args.shot), tar_label_bank_lbd)
    print('Feature saved!')


class Prototype:
    def __init__(self, C=65, dim=512, m=0.9):
        self.mo_pro = torch.zeros(C, dim).cuda()
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m

    @torch.no_grad()
    def update(self, feats, lbls, i_iter, norm=False):
        if i_iter < 20:
            momentum = 0
        else:
            momentum = self.m
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * \
                momentum + feats_i_center * (1 - momentum)
            self.batch_pro[i_cls, :] = feats_i_center
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)
            self.batch_pro = F.normalize(self.batch_pro)
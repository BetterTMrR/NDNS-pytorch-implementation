import ot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets, n_src=0):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        n_total = inputs.shape[0]
        epsilon = torch.cat([torch.ones((n_src, 1)), torch.zeros(n_total-n_src, 1)], dim=0).cuda().float()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        epsilon = self.epsilon * epsilon if n_src > 0 else self.epsilon
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


def loss_unl(net_G, net_F, imgs_tu_w, imgs_tu_s, proto_s, args, tar_feature_bank=None, tar_target_bank=None, target=None, mask2=None, gidx=None):
    '''The proposed losses for unlabeled target samples

    Parameters:
        net_G (network)    --The backbone
        net_F (network)    --The classifier (fc-l2norm-fc)
        imgs_tu_w (tensor) --Weakly augmented inputs
        imgs_tu_s (tensor) --Strongly augmented inputs
        proto_s (tensor)   --Source prototypes

    Return the three losses
    '''
    # forward
    feat_tu = net_G(torch.cat((imgs_tu_w, imgs_tu_s), dim=0).cuda())
    feat_tu_con, logits_tu = net_F(feat_tu, return_feat=True)
    feat_tu_w, feat_tu_s = feat_tu_con.chunk(2)
    logits_tu_w, logits_tu_s = logits_tu.chunk(2)

    # sample-wise consistency
    pseudo_label = torch.softmax(logits_tu_w.detach() * args.T2, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=1)
    consis_mask = max_probs.ge(args.th).float()
    L_pl = (F.cross_entropy(logits_tu_s, targets_u, reduction='none') * consis_mask).mean()

    # class-wise consistency
    prob_tu_w = torch.softmax(logits_tu_w, dim=1)
    prob_tu_s = torch.softmax(logits_tu_s, dim=1)
    L_con_cls = contras_cls(prob_tu_w, prob_tu_s)

    # alignment consistency
    L_ot = ot_loss(proto_s, feat_tu_w, feat_tu_s, args)

    if tar_feature_bank is not None:
        # neigh consistency
        '''
        features = G(data_t_un[0].cuda())
        softmax_out = F.softmax(F1(features, reverse=True, alpha=1), dim=1)
        output2 = F1(G(data_t_un[1].cuda()))
        logsoftmax_out = F.log_softmax(output2, dim=1)
        '''
        aen_loss = (F.kl_div(F.log_softmax(logits_tu_s, dim=1), target, reduction='none').sum(1) * mask2).mean()
        tar_feature_bank[gidx] = normalize(feat_tu_w).cpu().detach()
        tar_target_bank[gidx] = F.softmax(logits_tu_w, dim=1).cpu().detach()

        return L_ot, L_con_cls, L_pl, consis_mask, aen_loss
    else:
        return L_ot, L_con_cls, L_pl, consis_mask


def ot_loss(proto_s, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.class_num)
    return Lm


def ot_mapping(M):
    '''
    M: (ns, nt)
    '''
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_loss(proto_s, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.class_num)
    return Lm


def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat = -x @ centers_norm.t() + 1

    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss


def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def MME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, args):
    feat1 = G(inputs_lbd).cuda()
    outputs_lbd = F1(feat1)
    classifier_loss = SmoothLoss(outputs_lbd, labels_lbd, len(data_s[1])).mean()
    if args.lbd and (args.tar_aen or args.src_aen):
        tar_feature_bank[data_t[3]] = normalize(feat1[len(data_s[2]):].cpu().detach())
        tar_target_bank[data_t[3]] = F.softmax(outputs_lbd[len(data_s[2]):].cpu().detach(), dim=1)
    if args.src_aen:
        src_feature_bank[data_s[3]] = normalize(feat1[:len(data_s[2])].cpu().detach())
        src_target_bank[data_s[3]] = F.softmax(outputs_lbd[:len(data_s[2])].cpu().detach(), dim=1)

    if args.method != 'none':
        inputs_un = torch.cat((data_t_un[0], data_t_un[1]), dim=0).cuda()
        features_un = G(inputs_un)
        output_un = F1(features_un)
        output1, output2 = output_un.chunk(2)
        features, features_augm = features_un.chunk(2)
        softmax_out = F.softmax(output1, dim=1)
        if args.src_aen or args.tar_aen:
            logsoftmax_out = F.log_softmax(output2, dim=1)
            if args.tar_aen:
                pos_neigh_tt = topk_idx[data_t_un[4]]
                mask1 = edges[data_t_un[4]].cuda()
                pos_targets = tar_target_bank[pos_neigh_tt].cuda()
            if args.src_aen:
                pos_neigh_ts = idx_tar2src[data_t_un[4]]
                mask1 = edges_ts[data_t_un[4]].cuda()
                pos_targets = src_target_bank[pos_neigh_ts].cuda()
            if args.tar_aen and args.src_aen:
                pos_targets = torch.cat((tar_target_bank[pos_neigh_tt], src_target_bank[pos_neigh_ts]),
                                        dim=1).cuda()
                mask1 = torch.cat((edges[data_t_un[4]], edges_ts[data_t_un[4]]), dim=1).cuda()
            mask2 = mask1.sum(1) > 0
            target = (pos_targets * mask1.unsqueeze(-1)).sum(1) / (mask1.sum(1).reshape(-1, 1) + 1e-8)
            aen_loss = (F.kl_div(logsoftmax_out, target, reduction='none').sum(1) * mask2).mean()
            tar_feature_bank[data_t_un[4]] = normalize(features).cpu().detach()
            tar_target_bank[data_t_un[4]] = softmax_out.cpu().detach()
        else:
            aen_loss = torch.Tensor([0.0]).cuda()

        softmax_mme = F.softmax(F1(G(data_t_un[0].cuda()), reverse=True, alpha=1), dim=1)
        transfer_loss = classifier_loss + args.lam1 * aen_loss + args.lam * (
                    softmax_mme * torch.log(softmax_mme + 1e-8)).sum(1).mean()
        return transfer_loss


def FixMME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, args):
    feat1 = G(inputs_lbd).cuda()
    outputs_lbd = F1(feat1)
    classifier_loss = SmoothLoss(outputs_lbd, labels_lbd, len(data_s[1])).mean()
    if args.lbd and (args.tar_aen or args.src_aen):
        tar_feature_bank[data_t[3]] = normalize(feat1[len(data_s[2]):].cpu().detach())
        tar_target_bank[data_t[3]] = F.softmax(outputs_lbd[len(data_s[2]):].cpu().detach(), dim=1)
    if args.src_aen:
        src_feature_bank[data_s[3]] = normalize(feat1[:len(data_s[2])].cpu().detach())
        src_target_bank[data_s[3]] = F.softmax(outputs_lbd[:len(data_s[2])].cpu().detach(), dim=1)

    if args.method != 'none':
        inputs_un = torch.cat((data_t_un[0], data_t_un[1]), dim=0).cuda()
        features_un = G(inputs_un)
        output_un = F1(features_un)
        output1, output2 = output_un.chunk(2)
        features, features_augm = features_un.chunk(2)
        softmax_out = F.softmax(output1, dim=1)
        prob_tar, pred_tar = softmax_out.max(1)
        mask_tar = prob_tar.ge(args.th).float()
        fixmatch_loss = (SmoothLoss(output2, pred_tar) * mask_tar).mean()

        if args.src_aen or args.tar_aen:
            logsoftmax_out = F.log_softmax(output2, dim=1)
            if args.tar_aen:
                pos_neigh_tt = topk_idx[data_t_un[4]]
                mask1 = edges[data_t_un[4]].cuda()
                pos_targets = tar_target_bank[pos_neigh_tt].cuda()
            if args.src_aen:
                pos_neigh_ts = idx_tar2src[data_t_un[4]]
                mask1 = edges_ts[data_t_un[4]].cuda()
                pos_targets = src_target_bank[pos_neigh_ts].cuda()
            if args.tar_aen and args.src_aen:
                pos_targets = torch.cat((tar_target_bank[pos_neigh_tt], src_target_bank[pos_neigh_ts]),
                                        dim=1).cuda()
                mask1 = torch.cat((edges[data_t_un[4]], edges_ts[data_t_un[4]]), dim=1).cuda()
            mask2 = mask1.sum(1) > 0
            target = (pos_targets * mask1.unsqueeze(-1)).sum(1) / (mask1.sum(1).reshape(-1, 1) + 1e-8)
            aen_loss = (F.kl_div(logsoftmax_out, target, reduction='none').sum(1) * mask2).mean()
            tar_feature_bank[data_t_un[4]] = normalize(features).cpu().detach()
            tar_target_bank[data_t_un[4]] = softmax_out.cpu().detach()
        else:
            aen_loss = torch.Tensor([0.0]).cuda()

        softmax_mme = F.softmax(F1(G(data_t_un[0].cuda()), reverse=True, alpha=1), dim=1)
        transfer_loss = fixmatch_loss + classifier_loss + args.lam1 * aen_loss + args.lam * (
                    softmax_mme * torch.log(softmax_mme + 1e-8)).sum(1).mean()

        return transfer_loss


def MCL_loss(inputs_lbd, G, F1, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, lambda_warm, proto_s, args):
    feat1 = G(inputs_lbd).cuda()
    outputs_lbd = F1(feat1)
    classifier_loss = F.cross_entropy(outputs_lbd, labels_lbd)
    if args.lbd and (args.tar_aen or args.src_aen):
        tar_feature_bank[data_t[3]] = normalize(feat1[len(data_s[2]):].cpu().detach())
        tar_target_bank[data_t[3]] = F.softmax(outputs_lbd[len(data_s[2]):].cpu().detach(), dim=1)

    if args.src_aen:
        src_feature_bank[data_s[3]] = normalize(feat1[:len(data_s[2])].cpu().detach())
        src_target_bank[data_s[3]] = F.softmax(outputs_lbd[:len(data_s[2])].cpu().detach(), dim=1)

    if args.tar_aen:
        pos_neigh_tt = topk_idx[data_t_un[4]]
        mask1 = edges[data_t_un[4]].cuda()
        pos_targets = tar_target_bank[pos_neigh_tt].cuda()
    if args.src_aen:
        pos_neigh_ts = idx_tar2src[data_t_un[4]]
        mask1 = edges_ts[data_t_un[4]].cuda()
        pos_targets = src_target_bank[pos_neigh_ts].cuda()
    if args.tar_aen and args.src_aen:
        pos_targets = torch.cat((tar_target_bank[pos_neigh_tt], src_target_bank[pos_neigh_ts]),
                                dim=1).cuda()
        mask1 = torch.cat((edges[data_t_un[4]], edges_ts[data_t_un[4]]), dim=1).cuda()

    if args.tar_aen or args.src_aen:
        mask2 = mask1.sum(1) > 0
        target = (pos_targets * mask1.unsqueeze(-1)).sum(1) / (mask1.sum(1).reshape(-1, 1) + 1e-8)
        L_ot, L_con_cls, L_fix, consis_mask, aen_loss = loss_unl(G, F1, data_t_un[0], data_t_un[1], proto_s, args,
                                                                 tar_feature_bank, tar_target_bank, target, mask2,
                                                                 data_t_un[4])
        transfer_loss = classifier_loss + L_fix * args.lambda_u + L_con_cls * args.lambda_cls + lambda_warm * L_ot * args.lambda_ot + args.lam1 * aen_loss

    else:
        L_ot, L_con_cls, L_fix, consis_mask = loss_unl(G, F1, data_t_un[0], data_t_un[1], proto_s, args)
        transfer_loss = classifier_loss + L_fix * args.lambda_u + L_con_cls * args.lambda_cls + lambda_warm * L_ot * args.lambda_ot
    return transfer_loss, feat1
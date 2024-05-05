import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
from network import model_dict
from data_list import data_load_target
import random
import time
from active.sampler import ActiveStrategy
from loss_function import MCL_loss, MME_loss, FixMME_loss, CrossEntropyLabelSmooth
from utils import cal_acc, print_args, inv_lr_scheduler, memory_bank, \
    Prototype, adjacency_matrix


def ssl_training(args):
    dset_loaders = data_load_target(args)
    G, F1 = model_dict(args)
    args.lbd = True if args.ssda else False
    if not args.ssda:
        G.load_state_dict(torch.load(
            osp.join(args.output_dir_src, "source_{}_G.pt".format(args.seed))))
        F1.load_state_dict(torch.load(
            osp.join(args.output_dir_src, "source_{}_F.pt".format(args.seed))))
        G.eval()
        F1.eval()
        print('Prepare Memory Bank!')
        if args.src_aen:
            src_feature_bank, src_target_bank = memory_bank(dset_loaders['src_tr'], G, F1)
        if args.tar_aen:
            tar_feature_bank, tar_target_bank = memory_bank(dset_loaders['tar_un'], G, F1, metaidx=3)
        print('Finished!')

    G.train()
    F1.train()

    param_group_ssl = []
    param_group_ssl_f = []

    for k, v in G.named_parameters():
        if k[:4] == 'bott':
            param_group_ssl += [{'params': v, 'lr': 1}]
        else:
            param_group_ssl += [{'params': v, 'lr': 0.1}]

    for k, v in F1.named_parameters():
        param_group_ssl_f += [{'params': v, 'lr': 1}]

    optimizer_ssl = optim.SGD(param_group_ssl, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_ssl_f = optim.SGD(param_group_ssl_f, momentum=0.9, weight_decay=0.0005, nesterov=True)

    param_lr_g = []
    for param_group in optimizer_ssl.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_ssl_f.param_groups:
        param_lr_f.append(param_group["lr"])

    budget = args.shot * args.class_num
    active_num = [budget // args.rounds] * (args.rounds - 1)
    active_num.extend([budget - sum(active_num)])
    active_epoch = [(2 * (i + 1) - args.start) for i in range(len(active_num))]
    print(active_num)
    print(active_epoch)

    acc_best_val = 0.0
    epoch_num = 0
    iter_num = 0
    active_idx = 0

    interval_iter = len(dset_loaders["tar_un"])
    max_iter = max(50000, args.max_epoch * interval_iter)
    SmoothLoss = CrossEntropyLabelSmooth(reduction=False, num_classes=args.class_num, epsilon=args.epsilon)
    M1 = max(len(dset_loaders["tar_un"].dataset.exist) // (3 * args.class_num), 10)
    M2 = M1 // 5
    M1 = args.M1 if args.M1 > 0 else M1
    M2 = args.M2 if args.M2 > 0 else M2
    acc_list_test = []
    acc_list_val = []
    select_validation_set = True
    sampling_strategy = ActiveStrategy()
    start_time = time.time()
    proto_s = Prototype(C=args.class_num, dim=256)  # for MCL
    lambda_warm = 1  # for MCL
    while iter_num < max_iter:
        try:
            data_t_un = iter_tgt_un.next()
        except:
            iter_tgt_un = iter(dset_loaders["tar_un"])
            data_t_un = iter_tgt_un.next()
            if not args.ssda and args.tar_aen:
                topk_idx, edges = adjacency_matrix(tar_feature_bank, M1, M2)
        if args.lbd:
            try:
                data_t = iter_tgt.next()
            except:
                iter_tgt = iter(dset_loaders["tar_lbd"])
                data_t = iter_tgt.next()
        else:
            data_t = None
        try:
            data_s = iter_source.next()
        except:
            iter_source = iter(dset_loaders["src_tr"])
            data_s = iter_source.next()
            if not args.ssda and args.src_aen:
                tar_idx = torch.LongTensor([i for i in range(tar_feature_bank.size(0))])
                cosine_simi = src_feature_bank.mm(tar_feature_bank.t())  # [n_src, n_tar]
                _, idx_src2tar = cosine_simi.topk(dim=1, k=M2)   # [n_src, M2]
                _, idx_tar2src = cosine_simi.t().topk(dim=1, k=M2)   # [n_tar, M2]
                connection_ts = (idx_src2tar[idx_tar2src] == tar_idx[:, None, None]).sum(2)  # [n_tar, M2]
                edges_ts = torch.where(connection_ts > 0, torch.ones(connection_ts.size()), torch.zeros(connection_ts.size()))

        optimizer_ssl = inv_lr_scheduler(param_lr_g, optimizer_ssl, iter_num, init_lr=args.lr)
        optimizer_ssl_f = inv_lr_scheduler(param_lr_f, optimizer_ssl_f, iter_num, init_lr=args.lr)

        if args.lbd:
            inputs_lbd = torch.cat([data_s[0], data_t[0]], dim=0).cuda()
            labels_lbd = torch.cat([data_s[1], data_t[1]], dim=0).cuda()
        else:
            inputs_lbd = data_s[0].cuda()
            labels_lbd = data_s[1].cuda()

        if args.method == 'MME':
            if args.tar_aen and args.src_aen:
                transfer_loss = MME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, args)
            elif args.tar_aen:
                transfer_loss = MME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, None, None, topk_idx, edges, None, None, args)
            elif args.src_aen:
                transfer_loss = MME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, None, None, src_feature_bank, src_target_bank, None, None, idx_tar2src, edges_ts, args)
            else:
                transfer_loss = MME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, None, None, None, None, None, None, None, None, args)

        elif args.method == 'FixMME':
            if args.tar_aen and args.src_aen:
                transfer_loss = FixMME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, args)
            elif args.tar_aen:
                transfer_loss = FixMME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, None, None, topk_idx, edges, None, None, args)
            elif args.src_aen:
                transfer_loss = FixMME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, None, None, src_feature_bank, src_target_bank, None, None, idx_tar2src, edges_ts, args)
            else:
                transfer_loss = FixMME_loss(inputs_lbd, G, F1, SmoothLoss, labels_lbd, data_s, data_t, data_t_un, None, None, None, None, None, None, None, None, args)

        elif args.method == 'MCL':
            if args.tar_aen and args.src_aen:
                transfer_loss, feat1 = MCL_loss(inputs_lbd, G, F1, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, src_feature_bank, src_target_bank, topk_idx, edges, idx_tar2src, edges_ts, lambda_warm, proto_s, args)
            elif args.tar_aen:
                transfer_loss, feat1 = MCL_loss(inputs_lbd, G, F1, labels_lbd, data_s, data_t, data_t_un, tar_feature_bank, tar_target_bank, None, None, topk_idx, edges, None, None, lambda_warm, proto_s, args)
            elif args.src_aen:
                transfer_loss, feat1 = MCL_loss(inputs_lbd, G, F1, labels_lbd, data_s, data_t, data_t_un, None, None, src_feature_bank, src_target_bank, None, None, idx_tar2src, edges_ts, lambda_warm, proto_s, args)
            else:
                transfer_loss, feat1 = MCL_loss(inputs_lbd, G, F1, labels_lbd, data_s, data_t, data_t_un, None, None, None, None, None, None, None, None, lambda_warm, proto_s, args)
            proto_s.update(feat1[:len(data_s[2])], data_s[1], iter_num, norm=True)
        else:
            raise ValueError('Invalid SSDA method: {}.'.format(args.method))

        optimizer_ssl.zero_grad()
        optimizer_ssl_f.zero_grad()
        transfer_loss.backward()
        optimizer_ssl.step()
        optimizer_ssl_f.step()

        if iter_num % interval_iter == 0:
            if epoch_num in active_epoch and not args.ssda:
                if args.al_strategy == 'NDNS':
                    sampling_strategy.ndns_query(G, F1, active_num[active_idx], dset_loaders, False, args)
                active_idx += 1
            if epoch_num > active_epoch[-1] and select_validation_set:
                sampling_strategy.ndns_query(G, F1, 3*args.class_num, dset_loaders, True, args)
                select_validation_set = False

            epoch_num += 1

        iter_num += 1

        if iter_num % (args.interval * interval_iter) == 0 or iter_num == max_iter:
            G.eval()
            F1.eval()
            end_time = time.time()

            time_need = (end_time - start_time) * ((args.max_epoch - epoch_num) / args.interval)
            time_need = '{}h{}m{}s'.format(int(time_need // 3600), int((time_need % 3600) // 60), int(time_need % 60))
            time_cost = '{}m{}s. Time need: {}.'.format(int((end_time - start_time) // 60),
                                                                    int((end_time - start_time) % 60), time_need)
            if not select_validation_set:
                val_acc = cal_acc(dset_loaders['tar_val'], G, F1)
            else:
                val_acc = 0.0
            test_acc = cal_acc(dset_loaders['tar_test'], G, F1)

            if val_acc >= acc_best_val:
                acc_best_val = val_acc
                acc_best = test_acc

            acc_list_test.append(test_acc)
            acc_list_val.append(val_acc)
            log_str = '{}. Method: {}. Task: {} ({}). Epoch: [{}/{}]. Test Acc: ({:.1f}%) [({:.1f}%) (best)]. Val Acc: ({:.1f}%). Num: {}. Time: {}'.format(
               'SSDA' if args.ssda else 'Active', args.method, args.name, args.dset, epoch_num, args.max_epoch, test_acc, acc_best, val_acc, len(dset_loaders['tar_test'].dataset), time_cost)
            args.out_file_ssl.write(log_str + '\n')
            args.out_file_ssl.flush()
            print(log_str + '\n')
            G.train()
            F1.train()
            start_time = time.time()
        if epoch_num > args.max_epoch:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-maximal Degree Node Suppression')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source domain")
    parser.add_argument('--t', type=int, default=1, help="target domain")
    parser.add_argument('--hop', type=int, default=2, help="2-order neighbours")
    parser.add_argument('--rounds', type=int, default=6, help="rounds of active strategy")
    parser.add_argument('--max_epoch', type=int, default=40, help="training epochs")

    parser.add_argument('--T2', type=float, default=0.1, help="Hyperparameter for MCL")
    parser.add_argument('--lambda_cls', type=float, default=1., help="Hyperparameter for MCL")
    parser.add_argument('--lambda_u', type=float, default=1., help="Hyperparameter for MCL")
    parser.add_argument('--lambda_ot', type=float, default=1., help="Hyperparameter for MCL")
    parser.add_argument('--lam', type=float, default=0.1, help="Trade-off hyperparameter for MME")

    parser.add_argument('--M1', type=int, default=0, help="Number of ADNs")
    parser.add_argument('--M2', type=int, default=0, help="Number of AENs")
    parser.add_argument('--start', type=int, default=0, help="start to query")
    parser.add_argument('--interval', type=int, default=5, help="print interval")
    parser.add_argument('--ssda', action="store_true", help="whether use traditional SSDA method")
    parser.add_argument('--src_aen', action="store_true", help="whether use source domain AEN")
    parser.add_argument('--tar_aen', action="store_true", help="whether use target domain AEN")
    parser.add_argument('--lam1', type=float, default=0.3, help="trade-off hyperparameter")
    parser.add_argument('--alpha', type=float, default=0.1, help="trade-off for active strategy")
    parser.add_argument('--epsilon', type=float, default=0.1, help="label smoothing parameter")
    parser.add_argument('--th', type=float, default=0.8, help="hyperparameter for Fixmatch")
    parser.add_argument('--batch_size', type=int, default=48, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='office_home', choices=['office', 'office_home', 'multi'], help='data set')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet34', help="vgg16, resnet34, alexnet")
    parser.add_argument('--method', type=str, default='FixMME', help='SSDA method', choices=['MME', 'FixMME', 'MCL'])
    parser.add_argument('--al_strategy', type=str, default='NDNS', help='SSDA method')
    parser.add_argument('--seed', type=int, default=2024, help="random seed")
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--output', type=str, default='./record')
    parser.add_argument('--da', type=str, default='semiDA')
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.names = ['Art', 'Clipart', 'Product', 'Real']
        args.root = '/home/hejiujun/data/office_home/imgs/'  # root for images
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.names = ['amazon', 'dslr', 'webcam']
        args.root = '/home/hejiujun/data/office/domain_adaptation_images/'  # root for images
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.names = ['train', 'validation']
        args.class_num = 12
        args.root = '/userhome/home/hejiujun/data'
    if args.dset == 'multi':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.root = '/root/autodl-tmp/data/'  # root for images

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    alexnet = False
    if args.net == 'alexnet':
        print('reset batch size!')
        alexnet = True
        args.batch_size = 64

    folder = './data/source_txt/{}/'.format(args.dset)
    args.s_dset_path = folder + "labeled_images_{}.txt".format(names[args.s])
    args.t_dset_path = folder + "labeled_images_{}.txt".format(names[args.t])
    args.t_dset_path_un = folder + "unlabeled_target_images_{}_{}.txt".format(names[args.t], args.shot)
    args.t_dset_path_lbd = folder + "labeled_target_images_{}_{}.txt".format(names[args.t], args.shot)
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir_tgt = osp.join(args.output, args.da, args.dset, args.name, args.net)
    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper(), args.net)
    if not osp.exists(args.output_dir_tgt):
        os.system('mkdir -p ' + args.output_dir_tgt)
    if not osp.exists(args.output_dir_tgt):
        os.mkdir(args.output_dir_tgt)

    log_str = print_args(args)
    out_file_ssl_path = osp.join(args.output_dir_tgt,
                                      '{}_{}({})_shot:{}_alpha:{}_lam1:{}_(M1:{},M2:{})_(s:{},t:{})_{}.txt'.format(
                                          'SSDA' if args.ssda else 'Active',
                                          args.method,
                                          args.al_strategy,
                                          args.shot,
                                          args.alpha,
                                          args.lam1,
                                          args.M1,
                                          args.M2,
                                          args.src_aen,
                                          args.tar_aen,
                                          args.seed
                                      ))
    args.out_file_ssl = open(out_file_ssl_path, 'w')
    args.out_file_ssl.write(log_str + '\n')
    args.out_file_ssl.flush()
    ssl_training(args)

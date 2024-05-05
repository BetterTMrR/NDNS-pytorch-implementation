import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_list import ImageList_idx, image_train
from tqdm import tqdm


al_dict = {}


def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator


def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)


def adjacency_matrix(features, M1, M2):
    features = features / ((features ** 2).sum(1) ** 0.5).unsqueeze(1)
    global_idx = torch.LongTensor([i for i in range(features.size(0))])
    cosine_simi = features.mm(features.t())
    _, _topk_idx = cosine_simi.topk(dim=1, k=len(features))
    topk_idx = _topk_idx[:, 1:M1]
    connected = (topk_idx[:, :M2][topk_idx] == global_idx.unsqueeze(-1).unsqueeze(-1)).sum(2)
    edges = torch.where(connected > 0, torch.ones(connected.size()), torch.zeros(connected.size()))
    return topk_idx.detach(), edges.detach()


class ActiveStrategy:
    def __init__(self):
        pass

    def get_pred(self, G, F1):
        G.eval()
        F1.eval()
        all_label = torch.zeros(len(self.tar_loader.dataset), )
        all_fea = torch.zeros(len(self.tar_loader.dataset), 256)
        all_out = torch.zeros(len(self.tar_loader.dataset), F1.fc.out_features)
        all_idx = torch.zeros(len(self.tar_loader.dataset), )
        with torch.no_grad():
            iter_test = iter(self.tar_loader)
            i = 0
            for _ in tqdm(range(len(iter_test)), desc='Active Strategy'):
                data = iter_test.next()
                inputs = data[0]
                labels = data[2]
                idx = data[3]
                inputs = inputs.cuda()
                feat = G(inputs)
                out = F1(feat)
                all_label[i:i + len(labels)] = labels
                all_fea[i:i + len(labels)] = feat.cpu().detach()
                all_out[i:i + len(labels)] = out.cpu().detach()
                all_idx[i:i + len(labels)] = idx
                i += len(labels)
        print('Num. of all unlabeled target data: {}.'.format(len(all_label)))
        G.train()
        F1.train()
        return all_fea, all_out, all_label, all_idx.long().numpy()

    def query(self, G, F1, n, loader, args):
        self.tar_loader = loader["tar_un"]
        if args.al_strategy == 'badge':
            chosen = self.badge_query(G, F1, n)
        if args.al_strategy == 'las':
            chosen = self.las_query(G, F1, n)
        if args.al_strategy == 'clue':
            chosen = self.clue_query(G, F1, n)
        if args.al_strategy == 'mhp':
            chosen = self.mhp_query(G, F1, n, args)

        samples = loader["tar_un"].dataset.imgs, loader["tar_un"].dataset.labels
        exist = loader["tar_un"].dataset.exist[chosen]

        remove_item = [loader["tar_un"].dataset.mapping_idx2item[i] for i in chosen]
        remove_idx = np.array([loader["tar_test"].dataset.mapping_item2idx[item] for item in remove_item])

        # loader["tar_un"].dataset.global_marker[all_idx.numpy()[mark_idxes]] = -1
        loader["tar_un"].dataset.remove_item(chosen)
        loader["tar_test"].dataset.remove_item(remove_idx)

        try:
            loader['tar_lbd'].dataset.add_item(samples[0][chosen], samples[1][chosen], exist)
        except:
            print('There are no labeled data in the target dataset.')
            txt_lbd = [img + ' ' + label for img, label in
                       zip(samples[0][chosen], samples[1][chosen])]
            dataset_lbd = ImageList_idx(txt_lbd, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'),
                                        exist=exist)
            loader_lbd = DataLoader(dataset_lbd, batch_size=args.batch_size // 2, shuffle=True,
                                    num_workers=args.worker,
                                    drop_last=False)
            loader['tar_lbd'] = loader_lbd
            args.lbd = True
        lbd_items = samples[0][chosen]
        test_items = loader["tar_test"].dataset.imgs
        len_marked = loader["tar_un"].dataset.global_marker[loader["tar_un"].dataset.global_marker != 1].shape[0]
        len_total = len(loader["tar_un"].dataset.global_marker)

        print('Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked,
                                                                                           len_total, args.alpha))
        args.out_file_ssl.write(
            'Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked,
                                                                                         len_total, args.alpha) + '\n'
            )

        # check whether there exist labeled samples in the target test set
        num = 0
        for item in lbd_items:
            num += (item.reshape(-1, 1) == test_items.reshape(-1, 1)).sum()

        print('There are {} labeled instances in the test set.'.format(num))

        args.out_file_ssl.flush()

    def ndns_query(self, G, F1, active_num, loader, val, args):
        print('ndns_query')
        self.tar_loader = loader["tar_un"]
        all_fea, all_out, all_label, all_idx = self.get_pred(G, F1)
        all_idx = torch.from_numpy(all_idx).long()
        all_out = nn.Softmax(dim=-1)(all_out)
        prob_sort, _ = all_out.sort(dim=1, descending=True)
        global_marker = torch.from_numpy(loader["tar_un"].dataset.global_marker)[all_idx]
        M1 = max(len(all_idx) // (3 * all_out.shape[1]), 10)
        M2 = M1 // 5
        M1 = args.M1 if args.M1 > 0 else M1
        M2 = args.M2 if args.M2 > 0 else M2
        nearest_idxes, edges = adjacency_matrix(all_fea, M1, M2)
        chosen = []
        degree = edges.sum(1)
        degree /= (degree.max() + 1e-8)
        margin = prob_sort[:, 0] - prob_sort[:, 1]
        margin = (1 - margin) / (1 - margin).max()
        uncertainty = args.alpha * degree + margin
        uncertainty[global_marker != 1] = -1
        for _ in range(active_num):
            _, sorted_idxes = uncertainty.sort(descending=True)
            idx_chosen = sorted_idxes[0]
            chosen.append(idx_chosen)
            uncertainty[idx_chosen] = -1
            for __ in range(args.hop):
                connected_idx = nearest_idxes[idx_chosen][edges[idx_chosen] > 0]
                connected_idx = torch.from_numpy(np.unique(connected_idx.numpy())).long()
                uncertainty[connected_idx] = -1
                idx_chosen = connected_idx
        chosen = torch.LongTensor(chosen)
        mark_idxes = chosen.clone()

        # mask the 1-order neighbors of queried samples
        for _ in range(1):
            mark_idxes = nearest_idxes[mark_idxes][edges[mark_idxes] > 0]
            mark_idxes = torch.from_numpy(np.unique(mark_idxes.numpy())).long()
        mark_idxes = np.unique(mark_idxes.numpy()).astype('int64')
        samples = loader["tar_un"].dataset.imgs, loader["tar_un"].dataset.labels
        exist = loader["tar_un"].dataset.exist[all_idx[chosen].numpy()]

        remove_item = [loader["tar_un"].dataset.mapping_idx2item[i] for i in all_idx[chosen].numpy()]
        remove_idx = np.array([loader["tar_test"].dataset.mapping_item2idx[item] for item in remove_item])
        if not val:
            loader["tar_un"].dataset.global_marker[all_idx.numpy()[mark_idxes]] = -1
            loader["tar_un"].dataset.remove_item(all_idx[chosen].numpy())
            loader["tar_test"].dataset.remove_item(remove_idx)

        # select validation set
        if val:
            txt_lbd = [img + ' ' + label for img, label in
                       zip(samples[0][all_idx[chosen].numpy()], samples[1][all_idx[chosen].numpy()])]
            dataset_lbd = ImageList_idx(txt_lbd, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'),
                                        exist=exist)
            loader_lbd = DataLoader(dataset_lbd, batch_size=args.batch_size*5, shuffle=True,
                                    num_workers=args.worker,
                                    drop_last=False)
            loader['tar_val'] = loader_lbd
        else:
            try:
                loader['tar_lbd'].dataset.add_item(samples[0][all_idx[chosen].numpy()], samples[1][all_idx[chosen].numpy()],
                                                   exist)
            except:
                print('There are no labeled data in the target dataset.')
                txt_lbd = [img + ' ' + label for img, label in
                           zip(samples[0][all_idx[chosen].numpy()], samples[1][all_idx[chosen].numpy()])]
                dataset_lbd = ImageList_idx(txt_lbd, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'),
                                            exist=exist)
                loader_lbd = DataLoader(dataset_lbd, batch_size=args.batch_size // 2, shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
                loader['tar_lbd'] = loader_lbd
                args.lbd = True
        lbd_items = samples[0][all_idx[chosen].numpy()]
        test_items = loader["tar_test"].dataset.imgs
        len_marked = loader["tar_un"].dataset.global_marker[loader["tar_un"].dataset.global_marker != 1].shape[0]
        len_total = len(loader["tar_un"].dataset.global_marker)

        print('Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked,
                                                                                           len_total, args.alpha))
        args.out_file_ssl.write(
            'Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked,
                                                                                         len_total, args.alpha) + '\n'
            )

        # check whether there exist labeled samples in the target test set
        num = 0
        for item in lbd_items:
            num += (item.reshape(-1, 1) == test_items.reshape(-1, 1)).sum()

        print('There are {} labeled instances in the test set.'.format(num))

        args.out_file_ssl.flush()
        G.train()
        F1.train()

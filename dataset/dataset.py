import torch
import torch.nn
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from Sampler import MetaSampler
import torchnet as tnt
import torch.nn.functional as F
import copy
from collections import defaultdict
import random
import math
from sklearn import preprocessing
import sklearn.linear_model as models

class SUN:
    def __init__(self, dataroot='./dataset'):
        super(SUN, self).__init__()
        dataset = 'SUN'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T

        # scaler = preprocessing.MinMaxScaler()
        # feature = scaler.fit_transform(feature)
        # feature = F.normalize(torch.from_numpy(feature).float()).numpy()
        # feature = (feature - np.mean(feature)) / np.std(feature)

        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T
        # scaler = preprocessing.MinMaxScaler()
        # attribute = scaler.fit_transform(attribute)
        attribute = F.normalize(torch.from_numpy(attribute).float()).numpy()

        print(np.max(attribute), np.min(attribute))
        # print(attribute.shape)

        self.attr = torch.from_numpy(attribute).float()

        x = feature[trainval_loc]  # train_features
        train_label = label[trainval_loc].astype(int)  # train_label
        train_id = np.unique(train_label)  # test_id
        self.train_id = torch.from_numpy(train_id).long()
        self.train_attr_pro = torch.from_numpy(attribute[train_id]).float()  # train attributes

        self.train_x, self.train_y = torch.from_numpy(x).float(), torch.from_numpy(train_label).long()

        x_test = feature[test_unseen_loc]  # test_feature
        test_label = label[test_unseen_loc].astype(int)  # test_label
        x_test_seen = feature[test_seen_loc]  # test_seen_feature
        test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
        test_id = np.unique(test_label)  # test_id
        att_pro = attribute[test_id]  # test_attribute

        self.test_unseen_x, self.test_unseen_y = torch.from_numpy(x_test).float(), torch.from_numpy(test_label).long()
        self.test_seen_x, self.test_seen_y = torch.from_numpy(x_test_seen).float(), torch.from_numpy(test_label_seen).long()
        self.test_att_pro = torch.from_numpy(att_pro).float()
        self.test_id = torch.from_numpy(test_id).long()


class CUB:
    def __init__(self, dataroot='./dataset'):
        super(CUB, self).__init__()
        dataset = 'CUB'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T

        # scaler = preprocessing.MinMaxScaler()
        # feature = scaler.fit_transform(feature)
        feature = (feature - np.mean(feature)) / np.std(feature) # best norm
        print(np.max(feature), np.min(feature))

        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T
        attribute = sio.loadmat(dataroot + "/" + dataset + "/" + "sent_splits.mat")['att'].T
        # print(sent_attr['att'].shape)

        attribute = F.normalize(torch.from_numpy(attribute).float(), p=2).data.numpy()
        print(np.max(attribute), np.min(attribute))
        # print(attribute.shape)

        self.attr = torch.from_numpy(attribute).float()

        x = feature[trainval_loc]  # train_features
        train_label = label[trainval_loc].astype(int)  # train_label
        train_id = np.unique(train_label)  # test_id
        self.train_id = torch.from_numpy(train_id).long()
        self.train_attr_pro = torch.from_numpy(attribute[train_id]).float()  # train attributes

        self.train_x, self.train_y = torch.from_numpy(x).float(), torch.from_numpy(train_label).long()

        x_test = feature[test_unseen_loc]  # test_feature
        test_label = label[test_unseen_loc].astype(int)  # test_label
        x_test_seen = feature[test_seen_loc]  # test_seen_feature
        test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
        test_id = np.unique(test_label)  # test_id
        att_pro = attribute[test_id]  # test_attribute

        self.test_unseen_x, self.test_unseen_y = torch.from_numpy(x_test).float(), torch.from_numpy(test_label).long()
        self.test_seen_x, self.test_seen_y = torch.from_numpy(x_test_seen).float(), torch.from_numpy(test_label_seen).long()
        self.test_att_pro = torch.from_numpy(att_pro).float()
        self.test_id = torch.from_numpy(test_id).long()


class AWA1:
    def __init__(self, dataroot='./dataset'):
        super(AWA1, self).__init__()
        dataset = 'AWA1'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T
        # scaler = preprocessing.MinMaxScaler()
        # feature = scaler.fit_transform(feature)
        # feature = F.normalize(torch.from_numpy(feature).float(), p=2).data.numpy()
        # feature = (feature - np.mean(feature)) / np.std(feature)

        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T
        attribute = F.normalize(torch.from_numpy(attribute).float(), p=2).data.numpy()
        self.attr = torch.from_numpy(attribute).float()

        x = feature[trainval_loc]  # train_features
        train_label = label[trainval_loc].astype(int)  # train_label
        train_id = np.unique(train_label)  # test_id
        self.train_id = torch.from_numpy(train_id).long()
        self.train_attr_pro = torch.from_numpy(attribute[train_id]).float()  # train attributes

        self.train_x, self.train_y = torch.from_numpy(x).float(), torch.from_numpy(train_label).long()

        x_test = feature[test_unseen_loc]  # test_feature
        test_label = label[test_unseen_loc].astype(int)  # test_label
        x_test_seen = feature[test_seen_loc]  # test_seen_feature
        test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
        test_id = np.unique(test_label)  # test_id
        att_pro = attribute[test_id]  # test_attribute

        self.test_unseen_x, self.test_unseen_y = torch.from_numpy(x_test).float(), torch.from_numpy(test_label).long()
        self.test_seen_x, self.test_seen_y = torch.from_numpy(x_test_seen).float(), torch.from_numpy(test_label_seen).long()
        self.test_att_pro = torch.from_numpy(att_pro).float()
        self.test_id = torch.from_numpy(test_id).long()

class aPY:
    def __init__(self, dataroot='./dataset'):
        super(aPY, self).__init__()
        dataset = 'aPY'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T


        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T
        attribute = F.normalize(torch.from_numpy(attribute).float(), p=2).numpy()

        # LASSO = models.Ridge(alpha=1)
        # # LASSO = models.Lasso(alpha= 1)
        # LASSO.fit(attribute.transpose(), attribute.transpose())
        # similar = LASSO.coef_
        # similar[similar < 1e-3] = 0
        # tmp = np.sum(similar, axis=1)
        # tmp1 = np.tile(tmp, (similar.shape[1], 1)).transpose()
        # similar = similar / tmp1
        # self.sim = torch.from_numpy(similar).float()
        # print(self.sim.size())

        self.attr = torch.from_numpy(attribute).float()

        x = feature[trainval_loc]  # train_features
        train_label = label[trainval_loc].astype(int)  # train_label
        train_id = np.unique(train_label)  # test_id
        self.train_id = torch.from_numpy(train_id).long()
        self.train_attr_pro = torch.from_numpy(attribute[train_id]).float()  # train attributes

        self.train_x, self.train_y = torch.from_numpy(x).float(), torch.from_numpy(train_label).long()

        x_test = feature[test_unseen_loc]  # test_feature
        test_label = label[test_unseen_loc].astype(int)  # test_label
        x_test_seen = feature[test_seen_loc]  # test_seen_feature
        test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
        test_id = np.unique(test_label)  # test_id
        att_pro = attribute[test_id]  # test_attribute

        self.test_unseen_x, self.test_unseen_y = torch.from_numpy(x_test).float(), torch.from_numpy(test_label).long()
        self.test_seen_x, self.test_seen_y = torch.from_numpy(x_test_seen).float(), torch.from_numpy(test_label_seen).long()
        self.test_att_pro = torch.from_numpy(att_pro).float()
        self.test_id = torch.from_numpy(test_id).long()

        # print(self.sim[train_id][:, train_id].size(), self.sim[test_id][:, train_id].size())


class AWA2:
    def __init__(self, dataroot='./dataset'):
        super(AWA2, self).__init__()
        dataset = 'AWA2'
        image_embedding = 'res101'
        class_embedding = 'att'

        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
        feature = matcontent['features'].T

        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        attribute = matcontent['original_att'].T
        attribute = F.normalize(torch.from_numpy(attribute).float(), p=2).data.numpy()
        self.attr = torch.from_numpy(attribute).float()

        x = feature[trainval_loc]  # train_features
        train_label = label[trainval_loc].astype(int)  # train_label
        train_id = np.unique(train_label)  # test_id
        self.train_id = torch.from_numpy(train_id).long()
        self.train_attr_pro = torch.from_numpy(attribute[train_id]).float()  # train attributes

        self.train_x, self.train_y = torch.from_numpy(x).float(), torch.from_numpy(train_label).long()

        x_test = feature[test_unseen_loc]  # test_feature
        test_label = label[test_unseen_loc].astype(int)  # test_label
        x_test_seen = feature[test_seen_loc]  # test_seen_feature
        test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
        test_id = np.unique(test_label)  # test_id
        att_pro = attribute[test_id]  # test_attribute

        self.test_unseen_x, self.test_unseen_y = torch.from_numpy(x_test).float(), torch.from_numpy(test_label).long()
        self.test_seen_x, self.test_seen_y = torch.from_numpy(x_test_seen).float(), torch.from_numpy(test_label_seen).long()
        self.test_att_pro = torch.from_numpy(att_pro).float()
        self.test_id = torch.from_numpy(test_id).long()


class MyDataLoader(object):
    def __init__(self, config, dataset, mode='train'):
        self.mode = mode
        self.num_workers = config['num_workers']
        self.batch_size = config['test_batch_size']
        self.dataset = dataset
        self.k_shot = config['k_shot']
        self.n_way = config['n_way']
        self.k_query = config['k_query']

        if mode == 'train':
            self.data, self.label, self.attr = dataset.train_data, dataset.train_label, dataset.seen_attr
            # self.batch_size = config['test_batch_size']
            self.batch_size = 1
            self.label2index = defaultdict(list)
            for index, label in enumerate(self.label):
                self.label2index[label].append(index)
            self.iter_num = config['episode_num']
            # self.iter_num = len(self.data)
        elif mode == 'seen':
            self.data, self.label, self.attr = dataset.test_data_seen, dataset.test_label_seen, dataset.seen_attr
            self.batch_size = config['test_batch_size']
            self.iter_num = len(self.data)
        else:
            self.data, self.label, self.attr = dataset.test_data_unseen, dataset.test_label_unseen, dataset.unseen_attr
            self.batch_size = config['test_batch_size']
            self.iter_num = len(self.data)
        self.all_labels = list(np.unique(self.label))
        self.data, self.label, self.attr = np.array(self.data), np.array(self.label), np.array(self.attr)

    def get_train_task(self, idx):
        self.num_classes = len(self.all_labels)
        few_shot_train_batch = []
        few_shot_valid_batch = []
        if self.n_way <= self.num_classes:
            batch_few_shot_classes = random.sample(self.all_labels, self.n_way)
        else:
            batch_few_shot_classes = np.random.choice(self.all_labels, self.n_way, True).tolist()
        batch_few_shot_classes = np.array(batch_few_shot_classes)
        support_label, query_label, attr = [], [], []
        for i, cls in enumerate(batch_few_shot_classes):
            sample_indexes = random.sample(self.label2index[cls], self.k_shot + self.k_query)
            few_shot_train_batch.extend(sample_indexes[:self.k_shot])
            few_shot_valid_batch.extend(sample_indexes[self.k_shot:])
            support_label.extend([i] * self.k_shot)
            query_label.extend([i] * self.k_query)
            attr.append(self.attr[cls])
        few_shot_valid_batch = np.array(few_shot_valid_batch)
        few_shot_train_batch = np.array(few_shot_train_batch)
        support = (self.data[few_shot_train_batch], np.array(support_label), np.array(attr))
        query = (self.data[few_shot_valid_batch], np.array(query_label))
        # np.random.seed(idx)
        # np.random.shuffle(query[0])
        # np.random.seed(idx)
        # np.random.shuffle(query[1])
        return support, query


    def get_test_task(self, index):
        return self.data[index], self.label[index]

    def get_seen_proto(self):
        seen_proto = []
        for lab in self.all_labels:
            lab_index = self.label == lab
            seen_proto.append(np.mean(np.array(self.data[lab_index]), axis=0))
        return np.array(seen_proto), np.array(self.attr)


    def get_iterator(self):
        sampler = self.get_train_task if self.mode == 'train' else self.get_test_task
        # sampler = self.get_test_task
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.iter_num), load=sampler)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return data_loader

    def __call__(self):
        return self.get_iterator()


if __name__ == '__main__':
    cub = CUB(dataroot='../dataset')
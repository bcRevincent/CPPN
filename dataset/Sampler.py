import copy, random
import torch
import math
import numpy as np
from collections import defaultdict
from dataset import aPY
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class MetaSampler(object):

    def __init__(self, labels, class_per_it=5, num_shot=5, iter_num=1000):
        super(MetaSampler, self).__init__()
        self.length = len(labels)
        self.iter_num = iter_num
        self.labels = copy.deepcopy(labels)
        self.label2index = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.label2index[label].append(index)
        self.all_labels  = sorted( list(self.label2index.keys()) )
        print(self.all_labels)
        self.num_classes = len(self.all_labels)
        self.sample_per_class = num_shot
        self.classes_per_it   = class_per_it

    def __repr__(self):
        return ('{name}({iters:5d} iters, {sample_per_class:} ways, {classes_per_it:} shot, {num_classes:} classes)'.format(name=self.__class__.__name__, **self.__dict__))

    def __iter__(self):
        # yield a batch of indexes
        spc = self.sample_per_class
        cpi = self.classes_per_it
        for it in range(self.iter_num):
          # batch_size = spc * cpi
          few_shot_train_batch = []
          few_shot_valid_batch = []
          if cpi <= len(self.all_labels):
            batch_few_shot_classes = random.sample(self.all_labels, cpi)
          else:
            batch_few_shot_classes = np.random.choice(self.all_labels, cpi, True).tolist()
          for i, cls in enumerate(batch_few_shot_classes):
            if len(self.label2index[cls]) < spc:
                continue
            sample_indexes = random.sample(self.label2index[cls], spc)
            few_shot_train_batch.extend(sample_indexes)
          yield few_shot_train_batch

    def __len__(self):
        return self.iter_num


if __name__ == '__main__':
    data = aPY(dataroot='../dataset')
    TD = TensorDataset(data.train_x, data.train_y)
    sampler = MetaSampler(data.train_y.numpy())
    for x in sampler:
        print(data.train_x[x].size())
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
import os

np.random.seed(7)

# if __name__ == '__main__':
#     source = []
#     labels = []
#     label_dict = {}
#     hiera = defaultdict(set)
#     with open('dataset/DBpedia/dbp.taxnomy', 'r') as f:
#         label_dict['Root'] = -1
#         for line in f.readlines():
#             line = line.strip().split('\t')
#             for i in line[1:]:
#                 if i not in label_dict:
#                     label_dict[i] = len(label_dict) - 1
#                 hiera[label_dict[line[0]]].add(label_dict[i])
#         label_dict.pop('Root')
#         hiera.pop(-1)
#     value_dict = {i: v for v, i in label_dict.items()}
#     torch.save(value_dict, 'dataset/DBpedia/value_dict.pt')
#     torch.save(hiera, 'dataset/DBpedia/slot.pt')


if __name__ == '__main__':
    root_path = "dataset/DBpedia/"
    source = []
    labels = []
    label_dict = {}
    hiera = defaultdict(set)
    with open(os.path.join(root_path, "formatted_data", "label0.txt"), encoding='utf-8') as fp:
        label0_list = [line.strip() for line in list(fp)]
    with open(os.path.join(root_path, "formatted_data", "label1.txt"), encoding='utf-8') as fp:
        label1_list = [line.strip() for line in list(fp)]
    with open(os.path.join(root_path, "formatted_data", "label2.txt"), encoding='utf-8') as fp:
        label2_list = [line.strip() for line in list(fp)]
    all_label = label0_list + label1_list + label2_list
    with open('dataset/DBpedia/dbp.taxnomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = all_label.index(i)
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: v for v, i in label_dict.items()}
    torch.save(value_dict, 'dataset/DBpedia/value_dict.pt')
    torch.save(hiera, 'dataset/DBpedia/slot.pt')

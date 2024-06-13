from sklearn.manifold import TSNE
from time import time
import json
import pickle
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def set_plt(start_time, end_time,title):
    plt.title(f'{title}')
    # plt.legend(title='', loc='upper right')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks([])

shot_num = 4
# train_dataset = []
# with open("dataset/WebOfScience/wos_train.json", 'r') as fp:
#     for line in list(fp):
#         line = line.strip()
#         data = json.loads(line)
#         train_dataset.append([data['doc_token'], data['doc_label']])
# seed_num = json.load(open("dataset/WebOfScience/few-shot/seed_171-shot_"+str(shot_num)+".json",'r'))
# embedding_list = pickle.load(open('wos_pkl/_'+str(shot_num)+'shot_none_171_embed_doc_1.pkl',"rb"))
# data, label = [], []
# for i in range(len(seed_num)):
#     # if train_dataset[seed_num[i]][1][0] == "biochemistry":
#         data.append(embedding_list['embedding'][i][0])
#         label.append(train_dataset[seed_num[i]][1][0])
# data = np.array(data)
# label = np.array(label)

train_dataset = []
with open("dataset/WebOfScience/wos_test.json", 'r') as fp:
    for line in list(fp):
        line = line.strip()
        data = json.loads(line)
        train_dataset.append([data['doc_token'], data['doc_label']])
# seed_num = json.load(open("dataset/WebOfScience/few-shot/seed_171-shot_"+str(shot_num)+".json",'r'))
embedding_list = pickle.load(open('_'+str(shot_num)+'shot_none_171_embed_doc_0.pkl',"rb"))
data, label = [], []
for i in range(len(embedding_list['embedding'])):
    # if train_dataset[seed_num[i]][1][0] == "biochemistry":
        data.append(embedding_list['embedding'][i][1])
        label.append(train_dataset[i][1][0])
data = np.array(data)
label = np.array(label)

print('starting T-SNE process')
start_time = time()
data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
x_min, x_max = np.min(data, 0), np.max(data, 0)
data = (data - x_min) / (x_max - x_min)
df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
df.insert(loc=1, column='label', value=label)
end_time = time()
print('Finished')
sns.scatterplot(x='x', y='y', hue='label', s=10, palette="Set2", data=df, legend = False)
set_plt(start_time, end_time, 'Layer 2 MASK Embeddings')
plt.savefig('1.jpg', dpi=400)
plt.show()

print(1)

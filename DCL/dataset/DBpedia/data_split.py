import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import re


# def split_train_dev_test():
f = open('dbp_train.json', 'r')
data = f.readlines()
f.close()
id = [i for i in range(308407)]
np_data = np.array(data)
np.random.shuffle(id)
np_data = np_data[id]
train, val = train_test_split(np_data, test_size=0.011, random_state=0)
train = list(train)
val = list(val)
f = open('dbp_train_zy.json', 'w')
f.writelines(train)
f.close()
f = open('dbp_val_zy.json', 'w')
f.writelines(val)
f.close()

print(len(train), len(val))
    # return

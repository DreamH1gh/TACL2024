import json
from tqdm import tqdm
txt = open("Data.txt")
labeltree = {}
for line in tqdm(txt.readlines()[1:]):
    l = line.split("\t")
    l1 = l[3].strip()
    l2 = l[4].strip()
    if l1 in labeltree:
        if not l2 in labeltree[l1]:
            labeltree[l1].append(l2)
    else:
        labeltree[l1] = [l2]
json.dump(labeltree, open("label_tree.json","w"))
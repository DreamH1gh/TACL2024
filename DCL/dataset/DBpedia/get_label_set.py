import json

label_set = {}
layer = [[],[],[]]
input_path = '/home/chy/FastChat/dbp/dbp_test.json'
with open(input_path, "r", encoding='utf-8') as f:
    for eachline in f:
        train_data = json.loads(eachline)
        for i in range(len(train_data['doc_label'])):
            if i == 0:
                try:
                    if train_data['doc_label'][i] not in label_set['Root']:
                        label_set['Root'].append(train_data['doc_label'][i])
                except:
                    label_set['Root'] = [train_data['doc_label'][i]]
            else:
                try:
                    if train_data['doc_label'][i] not in label_set[train_data['doc_label'][i-1]]:
                        label_set[train_data['doc_label'][i-1]].append(train_data['doc_label'][i])
                except:
                    label_set[train_data['doc_label'][i-1]] = [train_data['doc_label'][i]]
# f = open('dbp.taxnomy', 'w')
# for i in label_set.keys():
#     line = [i]
#     line.extend(label_set[i])
#     line = '\t'.join(line) + '\n'
#     f.write(line)
# f.close()
for i in range(3):
    f = open('dataset/DBpedia/formatted_data/label'+str(i)+'.txt', 'w')
    for j in range(len(layer[i])):
        line = layer[i][j] + '\n'
        f.write(line)
    f.close()
print(1)
# json.dump(label_set, open("DBpedia/data/label_tree.json","w"))

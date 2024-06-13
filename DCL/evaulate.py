import datetime
import logging
from tqdm import tqdm
import os
import torch
import argparse
import pickle
import json
import openprompt
from openprompt.utils.reproduciblity import set_seed
from openprompt.prompts import SoftVerbalizer, ManualTemplate

from models.hierVerb import HierVerbPromptForClassification

from processor import PROCESSOR
from processor_des import PROCESSOR1

from util.utils import load_plm_from_config, print_info
from util.data_loader import SinglePathPromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

use_cuda = True



start_time = datetime.datetime.now()
parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='bert-base-uncased')
parser.add_argument("--result_file", type=str, default="few_shot_train.txt")

parser.add_argument("--multi_mask", type=int, default=1)

parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--shuffle", default=0, type=int)
parser.add_argument("--contrastive_logits", default=1, type=int)
parser.add_argument("--constraint_loss", default=0, type=int)
parser.add_argument("--cs_mode", default=0, type=int)
parser.add_argument("--dataset", default="dbp", type=str)
parser.add_argument("--eval_mode", default=0, type=int)
parser.add_argument("--use_hier_mean", default=1, type=int)
parser.add_argument("--freeze_plm", default=0, type=int)

parser.add_argument("--multi_label", default=0, type=int)
parser.add_argument("--multi_verb", default=1, type=int)

parser.add_argument("--use_scheduler1", default=1, type=int)
parser.add_argument("--use_scheduler2", default=1, type=int)

parser.add_argument("--constraint_alpha", default=-1, type=float)

parser.add_argument("--imbalanced_weight", default=True, type=bool)
parser.add_argument("--imbalanced_weight_reverse", default=True, type=bool)

parser.add_argument("--device", default=0, type=int)

parser.add_argument("--lm_training", default=1, type=int)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr2", default=1e-4, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_seq_lens", default=512, type=int, help="Max sequence length.")

parser.add_argument("--use_new_ct", default=1, type=int)
parser.add_argument("--contrastive_loss", default=0, type=int)
parser.add_argument("--contrastive_alpha", default=0.99, type=float)

parser.add_argument("--contrastive_level", default=1, type=int)
parser.add_argument("--use_dropout_sim", default=1, type=int)
parser.add_argument("--batch_size", default=5, type=int)

parser.add_argument("--use_withoutWrappedLM", default=False, type=bool)
parser.add_argument('--mean_verbalizer', default=True, type=bool)
parser.add_argument("--lm_alpha", default=0.999, type=float)

parser.add_argument("--shot", type=int, default=1)
parser.add_argument("--label_description", type=int, default=1)

parser.add_argument("--seed", type=int, default=171)
parser.add_argument("--plm_eval_mode", default=False)
parser.add_argument("--verbalizer", type=str, default="soft")

parser.add_argument("--template_id", default=0, type=int)

parser.add_argument("--not_manual", default=False, type=int)
parser.add_argument("--depth", default=2, type=int)

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)

parser.add_argument("--early_stop", default=10, type=int)

parser.add_argument("--eval_full", default=0, type=int)

args = parser.parse_args()
if args.device != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    device = torch.device("cuda:0")
    use_cuda = True
else:
    use_cuda = False
    device = torch.device("cpu")

if args.contrastive_loss == 0:
    args.contrastive_logits = 0
    args.use_dropout_sim = 0

if args.shuffle == 1:
    args.shuffle = True
else:
    args.shuffle = False
print_info(args)
processor = PROCESSOR[args.dataset](shot=args.shot, seed=args.seed)

if args.label_description:
    processor1 = PROCESSOR1[args.dataset](shot=args.shot, seed=args.seed)
    train_data = processor1.train_example
    dev_data = processor1.dev_example
    test_data = processor1.test_example
    # dataset
    dataset = {}
    dataset['train'] = processor1.train_example
    dataset['dev'] = processor1.dev_example
    dataset['test'] = processor1.test_example
else:
    train_data = processor.train_example
    dev_data = processor.dev_example
    test_data = processor.test_example
    # dataset
    dataset = {}
    dataset['train'] = processor.train_example
    dataset['dev'] = processor.dev_example
    dataset['test'] = processor.test_example
train_data = [[i.text_a, i.label] for i in train_data]
dev_data = [[i.text_a, i.label] for i in dev_data]
test_data = [[i.text_a, i.label] for i in test_data]
hier_mapping = processor.hier_mapping

retrieve_pred = pickle.load(open('dbp_best_samples/_1shot_none_550_1.pkl',"rb"))
# llm_pred = []
# with open('1shot_res.txt', 'r') as f:
#     for line in f.readlines():
#         llm_pred.append(line.strip('\n'))
llm_pred = json.load(open('1shot_res.json','r'))
predicted_onehot = []
for i in range(len(llm_pred)):
    one_hot_label = torch.zeros(len(processor.all_labels))
    llm_pred_text = llm_pred[i].split('-')
    if len(llm_pred_text)==3:
        try:
            similar_label_l2 = processor.label_list[-1].index(llm_pred_text[-1])
            if similar_label_l2 != retrieve_pred['labels'][i][2][2]:
                predicted_label=[hier_mapping[0][1][hier_mapping[1][1][similar_label_l2]], hier_mapping[1][1][similar_label_l2]+len(processor.label_list[0]), similar_label_l2+len(processor.label_list[0])+len(processor.label_list[1])]
            else:
                predicted_label=retrieve_pred['labels'][i][0]
        except:
            predicted_label=retrieve_pred['labels'][i][0]  
    else:
        predicted_label=retrieve_pred['labels'][i][0]
    one_hot_label[predicted_label] = 1
    predicted_onehot.append(one_hot_label)

true_onehot = []
for i in range(len(test_data)):
    one_hot_label = torch.zeros(len(processor.all_labels))
    label_l2 = test_data[i][-1]
    true_label = [hier_mapping[0][1][hier_mapping[1][1][label_l2]], hier_mapping[1][1][label_l2]+len(processor.label_list[0]), label_l2+len(processor.label_list[0])+len(processor.label_list[1])]
    one_hot_label[true_label] = 1
    true_onehot.append(one_hot_label)

epsilon = 1e-8
tp, fp, fn = torch.zeros(len(processor.all_labels)),torch.zeros(len(processor.all_labels)),torch.zeros(len(processor.all_labels))
for i in tqdm(range(len(predicted_onehot))):
    tp += torch.sum(predicted_onehot[i].unsqueeze(0)*true_onehot[i].unsqueeze(0), dim=0)
    fp += torch.sum(predicted_onehot[i].unsqueeze(0)*(1-true_onehot[i].unsqueeze(0)), dim=0)
    fn += torch.sum((1-predicted_onehot[i].unsqueeze(0))*true_onehot[i].unsqueeze(0), dim=0)

p = torch.sum(tp).item()/(torch.sum(tp).item()+torch.sum(fp).item()+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
r = torch.sum(tp).item()/(torch.sum(tp).item()+torch.sum(fn).item()+epsilon)
micro_f1 = 2*p*r/(p+r+epsilon)  
#macro-f1
p = tp/(tp+fp+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
r = tp/(tp+fn+epsilon)
macro_f1 = 2*p*r/(p+r+epsilon)
macro_f1 = torch.where(torch.isnan(macro_f1), torch.zeros_like(macro_f1), macro_f1)
total_micro_f1 = micro_f1
total_macro_f1 = torch.mean(macro_f1).item()

print(1)

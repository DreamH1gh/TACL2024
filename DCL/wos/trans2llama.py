import json

# with open('label_set_full.json', 'r', encoding='utf-8') as f:
#     label_set = json.load(f)

# fo = open("question_wos_label_full.jsonl","w")

# q_id = 1
# for l in label_set:
#     prompt = 'Please describe "' + l  + '" in 10 words:'
#     item = {
#         "question_id": q_id,
#         "question": [prompt]
#     }
#     q_id += 1
#     fo.write(json.dumps(item)+"\n")

"""
{
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "What can you do?"
      },
      {
        "from": "gpt",
        "value": "I can chat with you."
      }
    ]
  }
"""
train_set = []
with open('shot_data/1shot/train_888.json', 'r', encoding='utf-8') as f:
    for i,l in enumerate(f.readlines()):
        text = json.loads(l.strip())["content"]
        label = json.loads(l.strip())["label"]
        l1 = label.split(", label2:")[0][7:]
        l2 = label.split(", label2:")[1]
        target = f"{l1}-{l2}"
        item = {
            "id": f"identity_{i}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"{text}"
                },
                {
                    "from": "gpt",
                    "value": f"{target}"
                }
            ]
        }
        train_set.append(item)


json.dump(train_set, open("wos_train_conversation_1shot.json","w"))
    
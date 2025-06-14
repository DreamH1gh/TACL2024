# Detailed Prompts mentioned in the manucript

We request ChatGPT via OpenAI official API package [openai](https://platform.openai.com/docs/api-reference/introduction)

## Label Description Generation

To generate the label descriptions we use:
```
example_label = "machine learning"
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ' Please describe "' + example_label  + '" in 50 words:'},
            ]
        )
example_label_description = completion.choices[0].message['content']
```

## Prompts for ICL
Firstly, we define some known variables:
```
TestDoc                                     # test document
CurLabel                                    # predicted current level label
demoDoc1, demoDoc2, demoDoc3                # top 3 demostrations
Doc1curLabel, Doc2curLabel, Doc3curLabel    # current level labels of top 3 demostrations 
Doc1Label, Doc2Label, Doc3Label             # next level labels of top 3 demostrations
LabelSet                                    # candidate labels for prediction. The default of LabelSet is the union set of demostrations
```

### English Dataset
To process "Ours (Top3) + ICL" we use:
```
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1:\n\t Test:\n\t Text: {demoDoc1}\n\t Current Label: {Doc1curLabel}\n\t Candidate Label Set:{LabelSet}\n\t Output: {Doc1Label}\n\t \
                        Example 2:\n\t Test:\n\t Text: {demoDoc2}\n\t Current Label: {Doc2curLabel}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {demoDoc3}\n\t Current Label: {Doc3curLabel}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Current Label: {CurLabel}\n\t Candidate Label Set:{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```
To process "zero-shot ICL" we use:
```
LabelSet = [All the labels of datasets]
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Just answer the label.
                        Test:\n\t Text:\n{TestDoc}\n\t Current Label: {CurLabel} \n\t Candidate Label Set:{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

### Chinese Dataset
For Chinese Dataset we use ChatGLM.
To process "Ours (Top3) + ICL" we use:
```
prompt = f"你是一个擅长文档归类的人工智能助手，请根据我们提供的文档摘要，从候选标签中选择最合适的标签。以下是三个例子：\n\t \
            例子1: 摘要：{demoDoc1}\n\t 已有标签：{Doc1curLabel} 候选标签：{LabelSet}\n\t  预测标签: {Doc1Label}\n\t \
            例子2: 摘要：{demoDoc2}\n\t 已有标签：{Doc2curLabel} 候选标签：{LabelSet}\n\t  预测标签: {Doc2Label}\n\t \
            例子2: 摘要：{demoDoc3}\n\t 已有标签：{Doc3curLabel} 候选标签：{LabelSet}\n\t  预测标签: {Doc3Label}\n\t \
            测试: 摘要：{demoDoc1}\n\t 已有标签：{CurLabel} 候选标签：{LabelSet}\n\t  预测标签:"

```

To process "zero-shot ICL" we use:
```
LabelSet = [All the labels of datasets]
prompt = f"你是一个擅长文档归类的人工智能助手，请根据我们提供的文档摘要，从候选标签中选择最合适的标签。 摘要：{demoDoc1}\n\t 已有标签：{CurLabel} 候选标签：{LabelSet}\n\t  预测标签:"

```

## Others
To process "ICL w/o iterative", we remove “Current Label: {CurLabel} \n\t" and predict all label in one request:
```
Doc1Label = "Doc1Label1-Doc1Label2-Doc1Label3"
Doc2Label = "Doc2Label1-Doc2Label2-Doc2Label3"
Doc3Label = "Doc2Label1-Doc2Label2-Doc2Label3"

completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1: Test:\n\t Text: {demoDoc1}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc1Label}\n\t \
                        Example 2: Test:\n\t Text: {demoDoc2}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {demoDoc3}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Candidate Label Set{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

To process "ICL w/ random samples", we replace "Example 1,2,3“ with randomly selected ones and replace the candidate label set with all the labels in the dataset:
```
RandDemoDoc1, RandDemoDoc2, RandDemoDoc3 = [random samples]
LabelSet = [All the labels of datasets]

completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1: Test:\n\t Text: {RandDemoDoc1}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc1Label}\n\t \
                        Example 2: Test:\n\t Text: {RandDemoDoc2}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {RandDemoDoc3}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Candidate Label Set{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

To preces "ICL w/o top K samples", we replace the retrieved demostrations with randomly selected ones, and replace the candidate label set with all the labels in the dataset:
```
LabelSet = [All the labels of datasets]
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1: Test:\n\t Text: {RandomDemoDoc1}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc1Label}\n\t \
                        Example 2: Test:\n\t Text: {RandomDemoDoc2}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {RandomDemoDoc3}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Candidate Label Set{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

To preces "ICL w/o similar samples", we remove the demostrations while keep the candidate label set extracted from them:
```
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Candidate Label Set{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

To preces "ICL w/o pruning", we replace the candidate label set with all the labels in the dataset:
```
LabelSet = [All the labels of datasets]
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1: Test:\n\t Text: {demoDoc1}\n\t Current Label: {Doc1curLabel}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc1Label}\n\t \
                        Example 2: Test:\n\t Text: {demoDoc2}\n\t Current Label: {Doc2curLabel}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {demoDoc3}\n\t Current Label: {Doc3curLabel}\n\t Candidate Label Set{LabelSet}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Current Label: {CurLabel} \n\t Candidate Label Set{LabelSet}\n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```

To preces "ICL w/o candidate label set", we directly remove candidate label set in the prompt:
```
completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    f"Chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
                {"role": "system", "content": 
                    f"Your task is to select the most appropriate label for input text from the Candidate Label Set. Here are some similar demonstrations:\n\t \
                        Demonstrations:\n\t Example 1: Test:\n\t Text: {demoDoc1}\n\t Current Label: {Doc1curLabel}\n\t Output: {Doc1Label}\n\t \
                        Example 2: Test:\n\t Text: {demoDoc2}\n\t Current Label: {Doc2curLabel}\n\t Output: {Doc2Label}\n\t \
                        Example 3: Test:\n\t Text: {demoDoc3}\n\t Current Label: {Doc3curLabel}\n\t Output: {Doc3Label}\n\t \
                        Test:\n\t Text:\n{TestDoc}\n\t Current Label: {CurLabel} \n\t Output:"
                },
            ]
        )
predited_label = completion.choices[0].message['content']
```
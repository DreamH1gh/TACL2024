# Retrieval-Style ICL for or Few-shot Hierarchical Text Classification

source code of
- TACL [Retrieval-style In-Context Learning for Few-shot Hierarchical Text Classificatio]

The repository also contain the implementations of:
- [EPR](https://github.com/OhadRubin/EPR)

## Setup

setup environment 

```
conda create -n htc python=3.9
conda activate htc
pip install -r requirement.txt 
```
Step-1:
```
# Train the indexer with:
python train.py
```

Step-2:
```
# Run embedding.py to get the embeddings
python embedding.py
```

Step=3:
```
# Run topk.py to get the TOP k demonstrations for ICL
python topk.py
```
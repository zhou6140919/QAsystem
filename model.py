import torch
from transformers import BertModel, BertTokenizer
from data import text_dataset
datas = {phase: f'data/{phase}-v2.0.json' for phase in ['train', 'test', 'dev']}
datasets = {phase: text_dataset(datas[phase]) for phase in ['train', 'test', 'dev']}
for phase in ['train', 'test', 'dev']:
    datasets[phase].checkqa()
    datasets[phase].make_vocab()
    datasets[phase].removestopwords()
    docs = datasets[phase].fit_for_bert()
    print(docs[:10])
    print()
    print('done!')
    break


from transformers import BertModel, BertTokenizer
import json
from torchtext import datasets
from collections import Counter
import spacy
import torch
import numpy as np
import h5py
nlp = spacy.load('en_core_web_sm')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class text_dataset:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.questions = []
        self.answers = []
        self.vocab = set()
        self.all_tokens = []
        self.docs = []
        self.dict = {}
        with open(self.file_name, 'r') as f:
            row_data = f.read()
            row_data = json.loads(row_data)
            self.row_data = row_data
            title_size = 0
            while title_size < len(self.row_data['data']):
                title_data = self.row_data['data'][title_size]

                paragraphs_size = 0
                while paragraphs_size < len(title_data['paragraphs']):
                    paragraphs_data = title_data['paragraphs'][paragraphs_size]

                    qa_size = 0
                    while qa_size < len(paragraphs_data['qas']):
                        qa_data = paragraphs_data['qas'][qa_size]

                        if not qa_data['question']:
                            question_data_clean = qa_data['question'] + [' no question']
                        else:
                            question_data_clean = qa_data['question']
                        self.questions += [question_data_clean]

                        if not qa_data['answers']:
                            answers_data_clean = qa_data['answers'] + [' no answers']
                        else:
                            answers_data_clean = qa_data['answers']
                        self.answers += [answers_data_clean]

                        qa_size += 1
                    
                    paragraphs_size += 1
                
                title_size += 1
                
        self.questions = [question.lower() for question in self.questions]
        
        file1 = 'questions.json'
        file2 = 'answers.json'
        with open(file1, 'w') as f:
            json.dump(self.questions, f)
        with open(file2, 'w') as f:
            json.dump(self.answers, f)
        
    def checkqa(self):
        assert len(self.questions) == len(self.answers)
        return self.questions, self.answers

    def make_vocab(self):
        for question in self.questions:
            doc = list(tokenizer.tokenize(question))
            for token in doc:
                self.vocab.add(token)
                self.all_tokens += token + '\n'
        self.vocab = list(self.vocab)
        self.vocab.append('[CLS]')
        self.vocab.append('[SEP]')
        self.vocab = [token + '\n' for token in self.vocab]
        with open('vocab.txt', 'w') as  f:
            f.writelines(self.vocab)
        with open('all_tokens.txt', 'w') as f:
            f.writelines(self.all_tokens)
        
        for index, token in enumerate(self.vocab):
            self.dict[token] = index
        return self.vocab, self.all_tokens
    
    # def statistics(self):
    #     freq = Counter(self.all_tokens)
    #     value = freq.values()
        
    def removestopwords(self):
        for question in self.questions:
            tmp = '[CLS] ' + question + ' [SEP]'
            doc = list(nlp(tmp))
            filtered_sentence = []
            for token in doc:
                if token.is_stop != True and token.tag_ != 'PU':
                    filtered_sentence += token.text
            self.docs += filtered_sentence
        return self.docs

    def fit_for_bert(self):
        for i, doc in enumerate(self.docs):
            input_id, sent_id = self.make_tensor(i, doc)
            encoded_layer = self.get_encoded_layers(input_id, sent_id)
            sent_embedding = self.get_sent_embedding(doc, encoded_layer)
            array = np.array(sent_embedding)
            construct = np.vstack((construct, array))
        bert_vec = construct
        f = h5py.File('bert_vec.h5', 'w')
        f.create_dataset('bert_vec', data=bert_vec)
        f.close()
        return bert_vec

    
    def get_id(i, doc):
        sent_ids = [i] * len(doc)
        return sent_ids
    
    def get_index(self, doc):
        input_id = tokenizer.convert_tokens_to_ids(doc)
        return input_id

    def make_tensor(self, i, doc):
        input_id = torch.tensor(self.get_index(doc))
        sent_id = torch.tensor(self.get_id(i, doc))
        return input_id, sent_id

    def get_encoded_layers(input_id, sent_id):
        with torch.no_grad():
            encoded_layers, _ = model(input_id, sent_id)
        return (encoded_layers)
    
    def get_sent_embedding(doc, encoded_layers):
        token_embeddings = []
        for j, token in enumerate(doc):
            hidden_layers = []
            for i, layer in enumerate(encoded_layers):
                batch = 0
                
                vec = encoded_layers[i][batch][j]

                hidden_layers.append(vec)
            
            token_embeddings.append(hidden_layers)
        
        concat = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0)
                  for layer in token_embeddings]
        summed = [torch.sum(torch.stack(layer)[-4:], 0)
                  for layer in token_embeddings]
        sentence_embedding = torch.mean(encoded_layers[11], 1)

        return sentence_embedding






    

                        



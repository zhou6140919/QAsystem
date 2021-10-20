from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-distilled-squad")

model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad")

# from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# model = BertModel.from_pretrained("bert-base-uncased")


from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset
import pandas as pd

def tokenize_func(example):
    example['output_field'] = [int(i) for i in example['output_field']]
    return tokenizer(example['input_field'], truncation=True)

'''
def load_origin_data(filename):
    for line in pd.read_json(filename, lines=True):
        if(line['track']=="amazon-kdd-cup-24-shopping-knowledge-reasoning"):
'''            

def load_origin_data(filename):
    return pd.read_json(filename, lines=True)

checkpoint = 'distilbert-base-uncased'
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

print("model loaded and cached")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

DATA_FILENAME = './data/modified.json'
data_df = load_origin_data(DATA_FILENAME)
print(data_df)
data_df = data_df.astype(str)
# data_df = data_df.rename(columns={'input_field': 'question', 'output_field': 'answers'})

# raw_data = Dataset.from_json('./origin.json')
raw_data = Dataset.from_pandas(data_df)
tokens = raw_data.map(tokenize_func, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokens['output_field'][0])
print(type(tokens['output_field'][0]))

print("tokens:---------->")
print(tokens)
print('*'*100)

tokens = tokens.rename_column('input_field', 'question')
tokens = tokens.rename_column('output_field', 'label')

print("modifed tokens:---------->")
print(tokens)
print('*'*100)
# Parameters
# training_args = TrainingArguments("./models/meta-llama/Meta-Llama-3-8B-Instruct")
training_args = TrainingArguments(
    f"test-KDD",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
        model,
        training_args,
        train_dataset=tokens,
        eval_dataset=tokens,
        data_collator=data_collator,
        tokenizer=tokenizer,
        )
trainer.train()
trainer.save_model("KDD-trained")

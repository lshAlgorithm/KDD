from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset

def tokenize_func(exsample):
    return tokenizer(example['input_field'], truncation=True)

checkpoint = './models/meta-llama/Meta-Llama-3-8B-Instruct'
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

print("model loaded and cached")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

raw_data = Dataset.from_json('./data/formal_json.json') 
tokens = raw_data.map(tokenize_func, batch=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("tokens:---------->")
print(tokens)
print('*'*100)

# Parameters
training_args = TrainingArguments("./models/meta-llama/Meta-Llama-3-8B-Instruct")
trainer = Trainer(
        model, 
        training_args,
        train_dataset=tokens,
        eval_dataset=tokens,
        data_collator=data_collator,
        tokenizer=tokenizer,
        )
trainer.train()

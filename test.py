    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
    from datasets import Dataset

    model = UserModel()


    tokenizer = model.tokenizer
    
    raw_data = Dataset.from_json('./data/formal_json.json') 
    tokens = raw_data.map(tokenize_func, batch=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("tokens:---------->")
    print(tokens)

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


# Lora finetune result
1. Dataset: `yhx.json`
* parameters:
    * trainer's
```python
    args = TrainingArguments(
        output_dir="./models/fine_tune",
        per_device_train_batch_size=4,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
```
    * Lora's
```py
config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
```
* result:
```powershell
{'loss': 4.3165, 'grad_norm': 2.4375, 'learning_rate': 5.2380952380952384e-05, 'epoch': 1.29}
{'loss': 0.0727, 'grad_norm': 0.91796875, 'learning_rate': 4.7619047619047615e-06, 'epoch': 2.58}
{'train_runtime': 46.8093, 'train_samples_per_second': 7.819, 'train_steps_per_second': 0.449, 'train_loss': 2.0913701398032054, 'epoch': 2.71}
```

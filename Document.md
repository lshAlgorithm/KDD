# Lora finetune方案
```powershell
conda activate finetune # only used in finetune
python models/fintune.py # get CUDA out of memory
```
* next step:
    * use finetuned model in vllm's API `Lora_path`
    * generate more data

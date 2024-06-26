# My_GO Model
Add finetune to `vanilla_llama3_baseline` to form `My_GO` model
* The finetune training
    * under `finetune` env
    * Configuration
```powershell
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.9.5
pip install "transformers==4.40.0"
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.29.3
pip install datasets==2.19.0
pip install peft==0.10.0

MAX_JOBS=8 pip install flash-attn --no-build-isolation
```
> more to see in requirements.txt
* Testing
    * under `myenv` env
    * requirements are in `../requirements.txt`
    * run under directory `KDD` with basic command `python local_evaluation.py`
        * `--model` set the model to use, `Mygo` as default, you can also use `origin`
        * `--test` set dataset for evaluation, `origin` as default, you can also use `yhx` to use `yhx -o`.
* Result
    * Find in './result.md'
    * Final result after `finetune`
|Task|Accuracy|
|---|---|
|Task8|0.913|
|Task9|1.000|
|Task10|0.958|
|Avg.|0.957|
      

# Update logs
* [2024.6.16]Model's name changed to `Llama3_8B_Mygo`
* [2024.6.16]Add the printing of outputs from the Chatbot.
# Guide to Writing Your Own Models

## Model Code Organization
For a streamlined experience, we suggest placing the code for all your models within the `models` directory. This is a recommendation for organizational purposes, but it's not a strict requirement.

## Model Base Class
Your models should inherit from the `ShopBenchBaseModel` class found in [base_model.py](base_model.py). We provide an example model, `dummy_model.py`, to illustrate how you might structure your own model. Crucially, your model class must implement the `batch_predict` method.

## Configuring Your Model
To ensure your model is recognized and utilized correctly, please specify your model class name in the [`user_config.py`](user_config.py) file, by following the instructions in the inline comments.

## Model Inputs and Outputs

### Inputs
- `batch` (`Dict[str, Any]`): A batch of inputs as a dictionary, where the dictionary has the following key:
    - `prompt` (`List[str]`): `A list if prompts representing the tasks in a batch`
- `is_multiple_choice` (`bool`): This indicates whether the task is a multiple choice question.

### Outputs

The output from your model's `batch_predict` function should be a list of string responses for all the prompts in the input batch.
Depending on the task, each response could be:
- A single integer (in the range [0, 3]) for multiple choice tasks.
- A comma-separated list of integers for ranking tasks.
- A comma-separated list of named entities for Named Entity Recognition (NER) tasks.
- (unconstrained) generated response for the generation tasks

For more information on how these responses are processed, please see [parsers.py](../parsers.py).


**Note** that the `task_type` will not be explicitly provided to your model. However, the information about the `task_type` is implicitly available in the prompt provided.

## Internet Access
Your model will not have access to the internet during evaluation. As such, you'll need to include any necessary model weights directly in your repository before submission. Ensure that your Model class is self-contained and fully operational without internet access.


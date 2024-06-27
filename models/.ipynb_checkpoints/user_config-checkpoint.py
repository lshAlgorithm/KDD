# Importing DummyModel from the models package.
# The DummyModel class is located in the dummy_model.py file inside the 'models' directory.

from models.dummy_model import DummyModel
UserModel = DummyModel

# This line establishes an alias for the DummyModel class to be used within this script.
# Instead of directly using DummyModel everywhere in the code, we're assigning it to 'UserModel'.
# This approach allows for easier reference to your model class when evaluating your models,

# UserModel = DummyModel


# When implementing your own model please follow this pattern:
#
# from models.your_model import YourModel
#
# Replace 'your_model' with the name of your Python file containing the model class
# and 'YourModel' with the class name of your model.
#
# Finally, assign YourModel to UserModel as shown below to use it throughout your script.
#
# UserModel = YourModel


# For example, to use the Llama3 8B Instruct baseline, you can comment the lines below:
# please remember to download the model weights and checking them into the repository 
# before submitting

choice = 2

from transformers.models.llama.modeling_llama import LlamaForQuestionAnswering
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from models.vanilla_llama3_baseline import Llama3_8B_ZeroShotModel
from models.Llama3_8B_ZeroShotModel_Mygo import Llama3_8B_ZeroShotModel_Mygo
# Initializing a LLaMA llama-7b style configuration
checkpoint = "meta-llama/Meta-Llama-3-8B"
checkpoint = "models.vanilla_llama3_baseline.Llama3_8B_ZeroShotModel"
checkpoint = "./models/meta-llama/Meta-Llama-3-8B-Instruct"

# configuration = LlamaConfig.from_pretrained()
# print(configuration)
tokenizer = 1

if choice == 0:
    #model = LlamaForQuestionAnswering.from_pretrained(checkpoint, use_safetensors=True)
    #model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, use_safetensors=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_safetensors=True)
elif choice == 1:
    model = Llama3_8B_ZeroShotModel()
    # tokenizer = model.tokenizer # well, just a lesson, such syntax can only use for function as attribute not a variable
else:
    model = Llama3_8B_ZeroShotModel_Mygo()
    # # 加载tokenizer
    # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_safetensors=True)
    # print("********************")
    # # 加载模型
    # self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto",torch_dtype=torch.bfloat16)

    # # 加载lora权重
    # self.model = PeftModel.from_pretrained(model, model_id=lora_name, config=config)
    # # tokenizer = model.tokenizer

# Importing DummyModel from the models package.
# The DummyModel class is located in the dummy_model.py file inside the 'models' directory.

from shared_args import args
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

# from transformers.models.llama.modeling_llama import LlamaForQuestionAnswering
# from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from models.vanilla_llama3_baseline import Llama3_8B_ZeroShotModel
from models.Llama3_8B_Mygo import Llama3_8B_Mygo
# Initializing a LLaMA llama-7b style configuration

if args.model == 'automodel':
    checkpoint = "./models/meta-llama/Meta-Llama-3-8B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, use_safetensors=True)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_safetensors=True)
elif args.model == 'origin':
    model = Llama3_8B_ZeroShotModel()
else:
    model = Llama3_8B_Mygo()

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
from transformers import LlamaModel, LlamaConfig
from models.vanilla_llama3_baseline import Llama3_8B_ZeroShotModel
from models.Llama3_8B_ZeroShotModel_Mygo import Llama3_8B_ZeroShotModel_Mygo
# Initializing a LLaMA llama-7b style configuration
checkpoint = "meta-llama/Meta-Llama-3-8B"
checkpoint = "models.vanilla_llama3_baseline.Llama3_8B_ZeroShotModel"
checkpoint = "./models/meta-llama/Meta-Llama-3-8B-Instruct"

# configuration = LlamaConfig.from_pretrained()
# print(configuration)

if choice == 0:
    UserModel = LlamaForQuestionAnswering.from_pretrained(checkpoint)
elif choice == 1:
    UserModel = Llama3_8B_ZeroShotModel
else:
    UserModel = Llama3_8B_ZeroShotModel_Mygo

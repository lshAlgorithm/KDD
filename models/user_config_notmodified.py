# Importing DummyModel from the models package.
# The DummyModel class is located in the dummy_model.py file inside the 'models' directory.


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


#从models包中导入DummyModel
#DummyModel类位于“models”目录下的dummy_model.py文件中。

#从模型。dummy_model导入DummyModel

#这一行为DummyModel类建立了一个别名，在这个脚本中使用。
#我们不是直接在代码中到处使用DummyModel，而是将它分配给UserModel。
#这种方法可以在评估模型时更容易地引用模型类。

# UserModel = DummyModel


#当实现你自己的模型时，请遵循以下模式:
#
# from models。导入YourModel
#
#将'your_model'替换为包含模型类的Python文件的名称
#和'YourModel'与你的模型的类名。
#
#最后，将YourModel赋值给UserModel，如下所示，以便在整个脚本中使用它。
#
# UserModel = YourModel


#例如，要使用Llama3 8B指示基线，您可以注释以下行:
#请记住下载模型权重并将它们检入存储库
#提交前

from models.dummy_model import DummyModel
UserModel = DummyModel
from models.vanilla_llama3_baseline import Llama3_8B_ZeroShotModel
UserModel = Llama3_8B_ZeroShotModel

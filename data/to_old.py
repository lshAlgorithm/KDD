import os

f = open("./modified.json", 'w')
with open("./generated_data.json") as file:
    content = file.read()
    content = content.replace("},", "}")
    content = content.replace("[", '')
    content = content.replace(']', '')
    content = content.replace('} {', '}\n{')
    f.write(content)

f.close()

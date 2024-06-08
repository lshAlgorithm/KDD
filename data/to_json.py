import os

f = open("./formal_json.json", 'w')
with open("./development.json") as file:
    content = file.read()
    content = content.replace("}", "},")
    content = '[' + content + ']'
    # content = content.replace(']', '')
    # content = content.replace('} {', '}\n{')
    f.write(content)

f.close()

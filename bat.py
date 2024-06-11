import os

f = open('./models/meta-llama/Meta-Llama-3-8B-Instruct/adapter_config.json', 'r')
out = open('./models/meta-llama/Meta-Llama-3-8B-Instruct/adapter_config_new.json', 'w')

for line in f:
    line = line.replace("\'", "\"")
    out.write(line)

out.close()
f.close()

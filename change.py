path = './test.py'
f = open('./lsh_demo.py','w')

with open(path, 'r') as file:
    for line in file:
        if line == '':
            f.write('\n')
            continue
        line = line[4:]
        # print(line)
        f.write(line)

f.close()

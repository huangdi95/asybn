#########################################################################
# Author: Huang Di
# Mail: hd232508@163.com
# Created Time: Tue 04 Feb 2020 09:22:24 AM CST
#########################################################################
import matplotlib.pyplot as plt
import numpy as np
files = ['train-ac_resnet50.log', 'train-resnet50.log', 'train-ac_resnet50_normal.log']
for item in files:
    f = open(item, 'r',encoding='utf-8')
    f.seek(0)
    acc1 = []
    for lines in f:
        if 'Acc@1' in lines:
            acc1.append(float(lines.split()[2]))
    f.close()
    epoch = list(range(len(acc1)))
    print(item)
    plt.plot(epoch, acc1, label=item)
    plt.legend()
plt.xlabel("epoch")
plt.ylabel("acc1")
plt.savefig('fig.png')

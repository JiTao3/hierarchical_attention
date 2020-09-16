
import sys
import os
import numpy as np


sys.path.append(os.path.abspath(os.getcwd()))

from util.qerror import cal_q_error, print_qerror


with open("data/resutlv1.0-e1.txt", 'r') as f:
    lines = f.readlines()
    label_output = [line.split(' ') for line in lines]
    label = [float(label) for label, _ in label_output]
    output = [float(output) for _, output in label_output]


len(label)

qerror = [cal_q_error(predict, actually) for predict, actually in zip(output, label)]

print_qerror(q_error=qerror)

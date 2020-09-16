from typing import List
import numpy as np


def cal_q_error(predict, label, log=True):
    if log:
        predict = np.e**predict
        label = np.e**label
    if predict > label:
        q_error = predict / label
    else:
        q_error = label / predict
    return q_error


def print_qerror(q_error: List):
    print("max qerror: {:.4f}".format(max(q_error)))
    print("mean qerror: {:.4f}".format(np.mean(q_error)))
    print("media qerror: {:.4f}".format(np.median(q_error)))
    print("90th qerror: {:.4f}".format(np.percentile(q_error, 90)))
    print("95th qerror: {:.4f}".format(np.percentile(q_error, 95)))
    print("99th qerror: {:.4f}".format(np.percentile(q_error, 99)))

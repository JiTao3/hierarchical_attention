import time
import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.abspath(os.getcwd()))
# print(sys.path)

from util.plan_to_tree import Node, parse_dep_tree_text, tree_feature_label
from util.prase_tree2node_leaf import tree2NodeLeafmat


class PlanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.planTrees, self.maxchild = parse_dep_tree_text(folder_name=root_dir)
        self.trees_labels = [tree_feature_label(i) for i in self.planTrees]
        self.transform = transform

    def __len__(self):
        return len(self.planTrees)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # root + label
        tree, label = self.trees_labels[idx]
        nodemat, leafmat = tree2NodeLeafmat(tree)

        return (tree, nodemat, leafmat, torch.tensor(label, dtype=torch.double).reshape((1)))


def remove_signle_tree(root_dir, target_dir):
    planTrees, _ = parse_dep_tree_text(folder_name=root_dir)
    plan_dir = sorted(os.listdir(root_dir))
    for dir_name, tree in zip(plan_dir, planTrees):
        if tree.children:
            with open(os.path.join(root_dir, dir_name), "r") as read_f:
                lines = read_f.readlines()
            with open(os.path.join(target_dir, dir_name), "w") as write_f:
                write_f.writelines(lines)


def test_label():
    dataset = PlanDataset(root_dir="/home/jitao/hierarchical_attention/data/deep_plan")
    for i, data in enumerate(dataset):
        tree, nodemat, leafmat, label = data
        # print(label.shape)
        print(label)
        if np.isnan(label.numpy()):
            print("nan:", i)
        if np.isinf(label.numpy()):
            print("inf", i)


if __name__ == "__main__":
    # remove_signle_tree(
    #     # root_dir="/data1/jitao/dataset/cardinality/all_plan",
    #     root_dir="/data1/slm/datasets/JOB/synthetic",
    #     target_dir="/data1/jitao/dataset/cardinality/deep_plan",
    # )
    # pass
    # data = PlanDataset(root_dir="data/data2")
    test_label()

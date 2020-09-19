from typing import List
from collections import deque
import copy

import numpy as np
import torch

from util.plan_to_tree import Node, parse_dep_tree_text


def add_node_index(root: Node) -> Node:
    # add an index tu the tree to identify a node uniquely
    # so that we can jsutufy the ancenstral relationship between two node
    index = 1

    def add_index(root: Node):
        nonlocal index
        if not root:
            return -1

        root.index = index
        index += 1
        for child in root.children:
            add_index(child)

    add_index(root)
    return root


def is_ancestor(leaf: Node, node: Node) -> bool:
    # function to determine whether node is an ancester of leaf
    node_queue = deque([node])
    while node_queue:
        cnt_node = node_queue.popleft()
        for child in cnt_node.children:
            node_queue.append(child)
            if child.index == leaf.index:
                return True
    return False


def parse_tree2leaves_node(root: Node):
    leaf = []
    node = []

    def plan_tree_leaves_node(root: Node):
        # return the tree leaves and node list
        if root.children:
            node.append(root)
            for child in root.children:
                plan_tree_leaves_node(child)
        else:
            leaf.append(root)

    plan_tree_leaves_node(root)
    return leaf, node


def treeInterpolation(root: Node, leaf, node):
    # global FEATURE_LEN

    add_node_index(root)

    feature_len = leaf.shape[-1]
    leaf_order, node_order = parse_tree2leaves_node(root=root)

    tree_depth = len(node_order)
    tree_width = len(leaf_order)

    interpolation_vec = torch.zeros((tree_depth + 1, tree_width, feature_len), dtype=torch.double)

    for leaf_index in range(tree_width):
        interpolation_vec[tree_depth][leaf_index] = leaf[leaf_index]

    for leaf_index in range(tree_width):
        for node_index in range(tree_depth):
            if is_ancestor(leaf=leaf_order[leaf_index], node=node_order[node_index]):
                interpolation_vec[node_index][leaf_index] = node[node_index]
    hierarchical_embeddings_vec = hierarchical_embeddings(
        root=root, leaf_order=leaf_order, node_order=node_order, feature_len=feature_len
    )
    # print(torch.nonzero(hierarchical_embeddings_vec))
    # test_upward(interpolation_vec)
    return interpolation_vec + hierarchical_embeddings_vec


def vertical_deepth(node: Node, leaf: Node) -> int:
    deepth = 0
    node_queue = deque([node])
    # size = len(node_queue)
    while node_queue:
        size = len(node_queue)
        deepth += 1
        while size:
            cnt_node = node_queue.popleft()
            size -= 1
            for child in cnt_node.children:
                node_queue.append(child)
                if child.index == leaf.index:
                    return deepth


def horizontal_width(root: Node) -> int:
    # if only root it will return root
    leaf, _ = parse_tree2leaves_node(root=root)
    return len(leaf)


def hierarchical_embeddings(root: Node, leaf_order: List, node_order: List, feature_len: int):
    # global FEATURE_LEN

    tree_depth = len(node_order)
    tree_width = len(leaf_order)
    # feature_len =
    vertical_len = feature_len // 2
    horizontal_len = feature_len // 2
    hierarchical_emebdding_vec = torch.zeros(
        (tree_depth + 1, tree_width, feature_len), dtype=torch.double)
    for leaf_index in range(tree_width):
        for node_index in range(tree_depth):
            node = node_order[node_index]
            leaf = leaf_order[leaf_index]
            if is_ancestor(leaf=leaf, node=node):
                depth = vertical_deepth(node=node, leaf=leaf)
                width = horizontal_width(root=node)
                # need to check depth and width < horizonal_len
                assert depth < horizontal_len and width < vertical_len
                hierarchical_emebdding_vec[node_index][leaf_index][depth - 1] = 1.0
                hierarchical_emebdding_vec[node_index][leaf_index][horizontal_len + width - 1] = 1.0
    return hierarchical_emebdding_vec


def upward_ca(interpolation_vec):
    interpolation_vec_cp = copy.copy(interpolation_vec)
    tree_depth, tree_width, feature_len = interpolation_vec.shape
    upward_ca_vec = torch.zeros((tree_depth - 1, tree_width, feature_len), dtype=torch.double)
    for leaf_index in range(tree_width):
        for node_index in range(tree_depth - 1):
            if interpolation_vec_cp[node_index][leaf_index].detach().numpy().any():
                # if(torch.is_nonzero(interpolation_vec[node_index][leaf_index])):
                num_not_null = 1
                upward_ca_vec[node_index][leaf_index] = interpolation_vec[tree_depth - 1][leaf_index]
                for in_node_index in range(node_index, tree_depth - 1):
                    if interpolation_vec_cp[in_node_index][leaf_index].detach().numpy().any():
                        # if(torch.is_nonzero(interpolation_vec[in_node_index][leaf_index])):
                        upward_ca_vec[node_index][leaf_index] += interpolation_vec[in_node_index][leaf_index]
                        num_not_null += 1
                # print(num_not_null)
                upward_ca_vec[node_index][leaf_index] /= num_not_null
    # test_upward(upward_ca_vec)
    return upward_ca_vec


def weightedAggregationCoeffi(root: Node):
    leaf_order, node_order = parse_tree2leaves_node(root=root)

    tree_depth = len(node_order)
    tree_width = len(leaf_order)
    agg_coeffi = torch.zeros((tree_depth), dtype=torch.double)
    agg_coeffi += torch.tensor([tree_width], dtype=torch.double)

    leaves_nodes = [parse_tree2leaves_node(rot) for rot in node_order]
    tree_size = [len(leaves) + len(nodes) for leaves, nodes in leaves_nodes]

    agg_coeffi += torch.tensor(tree_size, dtype=torch.double)
    return 1 / agg_coeffi


# def weighted_aggregation(upward_ca_vec):
#     # upward ca vec with dim = node + 1 * leaf * d
#     dim = upward_ca_vec.shape[2]
#     no_zero = np.count_nonzero(upward_ca_vec, axis=(1, 2))/dim
#     upward_ca_sum = np.sum(upward_ca_vec, axis=1)

#     # no_zero * upward ca sum in each line

#     weighted_aggregation_vec = upward_ca_sum * np.expand_dims(no_zero, 1)
#     return weighted_aggregation_vec


def test_interpolation():
    plan_tree, max_children = parse_dep_tree_text(folder_name="./data")
    add_node_index(plan_tree[1])
    leaf_order, node_order = parse_tree2leaves_node(root=plan_tree[1])
    tree_depth = len(node_order)
    tree_width = len(leaf_order)
    print(tree_depth, tree_width)
    test_interpolation = np.zeros((tree_depth, tree_width), dtype=np.double)
    for leaf_index in range(tree_width):
        for node_index in range(tree_depth):
            if is_ancestor(leaf=leaf_order[leaf_index], node=node_order[node_index]):
                test_interpolation[node_index][leaf_index] = 1
    print(test_interpolation)


def test_upward(upward_ca_vec):
    test_upward_vec = torch.sum(upward_ca_vec, dim=-1)
    print(torch.nonzero(test_upward_vec))


def tree2NodeLeafmat(root: Node):
    global FEATURE_LEN

    leaf_order, node_order = parse_tree2leaves_node(root)
    node_mat = np.array([node.data for node in node_order], dtype=np.double)
    leaf_mat = np.array([leaf.data for leaf in leaf_order], dtype=np.double)
    nodemat, leafmat = (torch.from_numpy(node_mat).double(), torch.from_numpy(leaf_mat).double())
    return nodemat, leafmat


if __name__ == "__main__":
    # print(os.path.abspath('.'))
    plan_tree, max_children = parse_dep_tree_text(folder_name="./data")
    add_node_index(plan_tree[1])
    leaf_order, node_order = parse_tree2leaves_node(root=plan_tree[1])

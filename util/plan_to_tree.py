import os
import numpy as np

operators = [
    "Merge Join",
    "Hash",
    "Index Only Scan using title_pkey on title t",
    "Sort",
    "Seq Scan",
    "Index Scan using title_pkey on title t",
    "Materialize",
    "Nested Loop",
    "Hash Join",
]
columns = [
    "ci.movie_id",
    "t.id",
    "mi_idx.movie_id",
    "mi.movie_id",
    "mc.movie_id",
    "mk.movie_id",
]
scan_features = np.load("./model_parameter/test_scan_features_64.npy")


def extract_time(line):
    data = line.replace("->", "").lstrip().split("  ")[-1].split(" ")
    start_cost = data[0].split("..")[0].replace("(cost=", "")
    end_cost = data[0].split("..")[1]
    rows = data[1].replace("rows=", "")
    width = data[2].replace("width=", "").replace(")", "")
    a_start_cost = data[4].split("..")[0].replace("time=", "")
    a_end_cost = data[4].split("..")[1]
    a_rows = data[5].replace("rows=", "")
    return (
        float(start_cost),
        float(end_cost),
        float(rows),
        float(width),
        float(a_start_cost),
        float(a_end_cost),
        float(a_rows),
    )


def extract_operator(line):
    operator = line.replace("->", "").lstrip().split("  ")[0]
    if operator.startswith("Seq Scan"):
        operator = "Seq Scan"
    return operator, operator in operators


def extract_attributes(operator, line, feature_vec, i=None):
    operators = [
        "Merge Join",
        "Hash",
        "Index Only Scan using title_pkey on title t",
        "Sort",
        "Seq Scan",
        "Index Scan using title_pkey on title t",
        "Materialize",
        "Nested Loop",
        "Hash Join",
    ]
    columns = [
        "ci.movie_id",
        "t.id",
        "mi_idx.movie_id",
        "mi.movie_id",
        "mc.movie_id",
        "mk.movie_id",
    ]
    operators_count = len(operators)  # 9
    if operator in ["Hash", "Materialize", "Nested Loop"]:
        pass
    elif operator == "Merge Join":
        if "Cond" in line:
            for column in columns:
                if column in line:
                    feature_vec[columns.index(column) + operators_count] = 1.0
    elif operator == "Index Only Scan using title_pkey on title t":
        #         feature_vec[15:56] = scan_features[i]
        if "Cond" in line:
            feature_vec[columns.index("t.id") + operators_count] = 1.0
            for column in columns:
                if column in line:
                    feature_vec[columns.index(column) + operators_count] = 1.0
    elif operator == "Sort":
        for column in columns:
            if column in line:
                feature_vec[columns.index(column) + operators_count] = 1.0
    elif operator == "Index Scan using title_pkey on title t":
        #         feature_vec[15:56] = scan_features[i]
        if "Cond" in line:
            feature_vec[columns.index("t.id") + operators_count] = 1.0
            for column in columns:
                if column in line:
                    feature_vec[columns.index(column) + operators_count] = 1.0
    elif operator == "Hash Join":
        if "Cond" in line:
            for column in columns:
                if column in line:
                    feature_vec[columns.index(column) + operators_count] = 1.0
    elif operator == "Seq Scan":
        feature_vec[15:79] = scan_features[i]  # 64


"""Tree node class"""
class Node(object):
    def __init__(self, data, parent=None, index=-1):
        self.data = data
        self.children = []
        self.parent = parent
        self.index = index

    def add_child(self, obj):
        self.children.append(obj)

    def add_parent(self, obj):
        self.parent = obj

    def __str__(self, tabs=0):
        tab_spaces = str.join("", [" " for i in range(tabs)])
        return (
            tab_spaces
            + "+-- Node: "
            + str.join("|", self.data)
            + "\n"
            + str.join("\n", [child.__str__(tabs + 2) for child in self.children])
        )


def parse_dep_tree_text(folder_name="data"):
    scan_cnt = 0
    max_children = 0
    plan_trees = []
    feature_len = 9 + 6 + 7 + 64
    for each_plan in sorted(os.listdir(folder_name)):
        # print(each_plan)
        with open(os.path.join(folder_name, each_plan), "r") as f:
            lines = f.readlines()
            feature_vec = [0.0] * feature_len
            operator, in_operators = extract_operator(lines[0])
            if not in_operators:
                operator, in_operators = extract_operator(lines[1])
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                    lines[1]
                )
                j = 2
            else:
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                    lines[0]
                )
                j = 1
            feature_vec[feature_len - 7 : feature_len] = [
                start_cost,
                end_cost,
                rows,
                width,
                a_start_cost,
                a_end_cost,
                a_rows,
            ]
            feature_vec[operators.index(operator)] = 1.0
            if operator == "Seq Scan":
                extract_attributes(operator, lines[j], feature_vec, scan_cnt)
                scan_cnt += 1
                root_tokens = feature_vec
                current_node = Node(root_tokens)
                plan_trees.append(current_node)
                continue
            else:
                while "actual" not in lines[j] and "Plan" not in lines[j]:
                    extract_attributes(operator, lines[j], feature_vec)
                    j += 1
            root_tokens = feature_vec  # 所有吗
            current_node = Node(root_tokens)
            plan_trees.append(current_node)

            spaces = 0
            node_stack = []
            i = j
            while not lines[i].startswith("Planning time"):
                line = lines[i]
                i += 1
                if line.startswith("Planning time") or line.startswith(
                    "Execution time"
                ):
                    break
                elif line.strip() == "":
                    break
                elif "->" not in line:
                    continue
                else:
                    if line.index("->") < spaces:
                        while line.index("->") < spaces:
                            current_node, spaces = node_stack.pop()

                    if line.index("->") > spaces:
                        line_copy = line
                        feature_vec = [0.0] * feature_len
                        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                            line_copy
                        )
                        feature_vec[feature_len - 7 : feature_len] = [
                            start_cost,
                            end_cost,
                            rows,
                            width,
                            a_start_cost,
                            a_end_cost,
                            a_rows,
                        ]
                        operator, in_operators = extract_operator(line_copy)
                        feature_vec[operators.index(operator)] = 1.0
                        if operator == "Seq Scan":
                            extract_attributes(
                                operator, line_copy, feature_vec, scan_cnt
                            )
                            scan_cnt += 1
                        else:
                            j = 0
                            while (
                                "actual" not in lines[i + j]
                                and "Plan" not in lines[i + j]
                            ):
                                extract_attributes(operator, lines[i + j], feature_vec)
                                j += 1
                        tokens = feature_vec
                        new_node = Node(tokens, parent=current_node)
                        current_node.add_child(new_node)
                        if len(current_node.children) > max_children:
                            max_children = len(current_node.children)
                        node_stack.append((current_node, spaces))
                        current_node = new_node
                        spaces = line.index("->")
                    elif line.index("->") == spaces:
                        line_copy = line
                        feature_vec = [0.0] * feature_len
                        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                            line_copy
                        )
                        feature_vec[feature_len - 7 : feature_len] = [
                            start_cost,
                            end_cost,
                            rows,
                            width,
                            a_start_cost,
                            a_end_cost,
                            a_rows,
                        ]
                        operator, in_operators = extract_operator(line_copy)
                        feature_vec[operators.index(operator)] = 1.0
                        if operator == "Seq Scan":
                            extract_attributes(
                                operator, line_copy, feature_vec, scan_cnt
                            )
                            scan_cnt += 1
                        else:
                            j = 0
                            while (
                                "actual" not in lines[i + j]
                                and "Plan" not in lines[i + j]
                            ):
                                extract_attributes(operator, lines[i + j], feature_vec)
                                j += 1
                        tokens = feature_vec
                        new_node = Node(tokens, parent=node_stack[-1][0])
                        node_stack[-1][0].add_child(new_node)
                        if len(node_stack[-1][0].children) > max_children:
                            max_children = len(node_stack[-1][0].children)
                        current_node = new_node
                        spaces = line.index("->")
    #         break
    # print(scan_cnt)
    return plan_trees, max_children  # a list of the roots nodes


def parse_dep_tree_text_lb_ub(folder_name="data/"):
    scan_cnt = 0
    max_children = 0
    plan_trees = []
    feature_len = 9 + 6 + 7 + 32
    for each_plan in sorted(os.listdir(folder_name)):
        #         print(each_plan)
        with open(os.path.join(folder_name, each_plan), "r") as f:
            lines = f.readlines()
            feature_vec = [0.0] * feature_len
            operator, in_operators = extract_operator(lines[0])
            if not in_operators:
                operator, in_operators = extract_operator(lines[1])
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                    lines[1]
                )
                j = 2
            else:
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                    lines[0]
                )
                j = 1
            feature_vec[feature_len - 7 : feature_len] = [
                start_cost,
                end_cost,
                rows,
                width,
                a_start_cost,
                a_end_cost,
                a_rows,
            ]
            feature_vec[operators.index(operator)] = 1.0
            if operator == "Seq Scan":
                extract_attributes(operator, lines[j], feature_vec, scan_cnt)
                scan_cnt += 1
                root_tokens = feature_vec
                current_node = Node(root_tokens)
                plan_trees.append(current_node)
                continue
            else:
                while "actual" not in lines[j] and "Plan" not in lines[j]:
                    extract_attributes(operator, lines[j], feature_vec)
                    j += 1
            root_tokens = feature_vec  # 所有吗
            current_node = Node(root_tokens)
            plan_trees.append(current_node)

            spaces = 0
            node_stack = []
            i = j
            while not lines[i].startswith("Planning time"):
                line = lines[i]
                i += 1
                if line.startswith("Planning time") or line.startswith(
                    "Execution time"
                ):
                    break
                elif line.strip() == "":
                    break
                elif "->" not in line:
                    continue
                else:
                    if line.index("->") < spaces:
                        while line.index("->") < spaces:
                            current_node, spaces = node_stack.pop()

                    if line.index("->") > spaces:
                        line_copy = line
                        feature_vec = [0.0] * feature_len
                        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                            line_copy
                        )
                        feature_vec[feature_len - 7 : feature_len] = [
                            start_cost,
                            end_cost,
                            rows,
                            width,
                            a_start_cost,
                            a_end_cost,
                            a_rows,
                        ]
                        operator, in_operators = extract_operator(line_copy)
                        feature_vec[operators.index(operator)] = 1.0
                        if operator == "Seq Scan":

                            # if(operator == "Seq Scan" or operator == "Index Only Scan using title_pkey on title t" or operator=='Index Scan using title_pkey on title t'):
                            extract_attributes(
                                operator, line_copy, feature_vec, scan_cnt
                            )
                            scan_cnt += 1
                        else:
                            j = 0
                            while (
                                "actual" not in lines[i + j]
                                and "Plan" not in lines[i + j]
                            ):
                                extract_attributes(operator, lines[i + j], feature_vec)
                                j += 1
                        tokens = feature_vec
                        new_node = Node(tokens, parent=current_node)
                        current_node.add_child(new_node)
                        if len(current_node.children) > max_children:
                            max_children = len(current_node.children)
                        node_stack.append((current_node, spaces))
                        current_node = new_node
                        spaces = line.index("->")
                    elif line.index("->") == spaces:
                        line_copy = line
                        feature_vec = [0.0] * feature_len
                        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_time(
                            line_copy
                        )
                        feature_vec[feature_len - 7 : feature_len] = [
                            start_cost,
                            end_cost,
                            rows,
                            width,
                            a_start_cost,
                            a_end_cost,
                            a_rows,
                        ]
                        operator, in_operators = extract_operator(line_copy)
                        feature_vec[operators.index(operator)] = 1.0
                        if operator == "Seq Scan":
                            # if(operator == "Seq Scan" or operator == "Index Only Scan using title_pkey on title t" or operator=='Index Scan using title_pkey on title t'):
                            extract_attributes(
                                operator, line_copy, feature_vec, scan_cnt
                            )
                            scan_cnt += 1
                        else:
                            j = 0
                            while (
                                "actual" not in lines[i + j]
                                and "Plan" not in lines[i + j]
                            ):
                                extract_attributes(operator, lines[i + j], feature_vec)
                                j += 1
                        tokens = feature_vec
                        new_node = Node(tokens, parent=node_stack[-1][0])
                        node_stack[-1][0].add_child(new_node)
                        if len(node_stack[-1][0].children) > max_children:
                            max_children = len(node_stack[-1][0].children)
                        current_node = new_node
                        spaces = line.index("->")
        # break
    # print(scan_cnt)
    return plan_trees, max_children  # a list of the roots nodes


def p2t(node):
    # prediction to true cardinality
    #         return float(start_cost),float(end_cost),float(rows),float(width),
    #         float(a_start_cost),float(a_end_cost),float(a_rows)
    tree = {}
    tmp = node.data
    operators_count = 9
    columns_count = 6
    scan_features = 64
    assert len(tmp) == operators_count + columns_count + 7 + scan_features
    tree["features"] = tmp[: operators_count + columns_count + scan_features]
    #     tree['features'].append(tmp[-5])  #with card as feature
    tree["features"].append(tmp[-1])  # with Actual card as feature
    # cardinality
    #     tree['labels'] = np.log(node.data[-1]+1) #cardinality
    #     tree['pg'] = np.log(node.data[-5])
    # cost
    tree["labels"] = np.log(node.data[-2])  # cost
    tree["pg"] = np.log(node.data[-6])


    tree["children"] = []
    for children in node.children:
        tree["children"].append(p2t(children))
    return tree

def tree_feature_label(root:Node):
    label = root.data[-1]
    operators_count = 9
    columns_count = 6
    scan_features = 64
    feature_len = operators_count + columns_count + scan_features
    def feature(root:Node):
        root.data = root.data[:feature_len]
        if root.children:
            for child in root.children:
                feature(child)
    return root, np.log(label)


if __name__ == "__main__":
    print(os.path.abspath("."))
    plan_tree, max_children = parse_dep_tree_text(folder_name="./data")
    # add_node_index(plan_tree[1])
    # leaf,node = test(plan_tree[1])

    print(len(plan_tree))

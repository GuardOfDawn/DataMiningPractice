import pandas as pd


class TreeNode:
    def __init__(self, name_value, num_occur, parent_node):
        self.value = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = []

    def add_children(self, c):
        self.children.append(c)

    def inc(self, num=1):
        self.count += num


def sort_according_to_frequency(to_sort, items):
    for a in range(0, len(to_sort)-1):
        for b in range(0, len(to_sort)-a-1):
            if items[to_sort[b]].count < items[to_sort[b+1]].count:
                tmp = to_sort[b]
                to_sort[b] = to_sort[b+1]
                to_sort[b+1] = tmp
    return to_sort


def visit_node(node, level):
    for t in node.children:
        visit_node(t, level+1)
    print(node.value+":"+str(level)+":"+str(node.count))


def create_tree(data_trans):
    items = {}
    for tran in data_trans:
        for an_item in tran:
            if items.__contains__(an_item):
                items[an_item].inc(data_trans[tran])
            else:
                items[an_item] = TreeNode(an_item, data_trans[tran], None)
    if len(items) == 0:
        return None, None
    # construct the FP-Tree
    root = TreeNode('Null Set', 1, None)
    for i in data_trans:
        tran = sort_according_to_frequency(list(i), items)
        tmp_node = root
        to_continue = True
        position = 0
        # visit existent nodes
        while to_continue:
            to_continue = False
            for n in tmp_node.children:
                if n.value == tran[position]:
                    to_continue = True
                    n.inc(data_trans[i])
                    position += 1
                    if position > len(tran)-1:
                        to_continue = False
                    tmp_node = n
                    break
        # add remaining nodes
        for j in range(position, len(tran)):
            child = TreeNode(tran[j], data_trans[i], tmp_node)
            tmp_node.add_children(child)
            # add link
            link_node = items[tran[j]]
            while link_node.node_link is not None:
                link_node = link_node.node_link
            link_node.node_link = child
            tmp_node = child
    return root, items


def mining_tree(in_tree, items, pre_fix, freq_items, min_support):
    # sorted_items_list is a list
    sorted_items_list = sorted(items.items(), key=lambda d: d[1].count, reverse=False)
    for an_item in sorted_items_list:
        new_freq_set = pre_fix.copy()
        new_freq_set.add(an_item[0])
        if len(new_freq_set) > 0 and an_item[1].count > min_support:
            freq_items[frozenset(new_freq_set)] = an_item[1].count
        node_cur = an_item[1]
        conditions = {}
        while node_cur.node_link is not None:
            node_cur = node_cur.node_link
            node_up = node_cur
            condition = []
            while (node_up.parent is not None) and (node_up.parent.value != 'Null Set'):
                condition.append(node_up.parent.value)
                node_up = node_up.parent
            if len(condition) > 0:
                conditions[frozenset(condition)] = node_cur.count
        my_cond_tree, my_items = create_tree(conditions)
        if my_items is not None:
            mining_tree(my_cond_tree, my_items, new_freq_set, freq_items, min_support)


df = pd.read_csv('F:\zhangyi\course\data mining\HR_attribute_reprocess.csv')

trans = {}
A = 'A'
for row in range(0, len(df.index)):
    transaction = []
    for col in range(0, len(df.columns)):
        label = str(chr(ord(A)+col)) + str(df.loc[row, df.columns[col]])
        transaction.append(label)
    trans[frozenset(transaction)] = trans.get(frozenset(transaction), 0)+1

columns = df.columns

total_count = len(df.index)
print(total_count)
satisfaction_list = ['B4']
left = 'J1'
min_sup = 0
min_confidence = 0.4
frequent_items = {}
a, b = data = create_tree(trans)
mining_tree(a, b, set([]), frequent_items, min_sup)
frequent_satisfaction = {}
for item in frequent_items:
    count = frequent_items[item]
    item = list(item)
    if 2 < len(item) < 4:
        for sat_item in satisfaction_list:
            if sat_item in item and left in item:
                item.remove(sat_item)
                item.remove(left)
                # confidence = count/(frequent_items[frozenset(item)]+0.0)
                # if confidence > min_confidence:
                #     key = columns[ord(item[0][0])-ord(A)]+item[0][1]
                #     for i in range(1, len(item)):
                #         key += ','
                #         key += columns[ord(item[i][0])-ord(A)]+item[i][1]
                #     key += "=>"+columns[1]+sat_item[1]+","+left
                #     frequent_satisfaction[key] = str(format(count/total_count, '.00%'))+" ; "+str(format(confidence, '.00%'))
                list2 = [sat_item, left]
                confidence2 = count/(frequent_items[frozenset(list2)]+0.0)
                if confidence2 > min_confidence:
                    key2 = sat_item+","+left  # columns[1]+sat_item[1]+","+left
                    key2 += "=>" + item[0]  # columns[ord(item[0][0])-ord(A)]+item[0][1]
                    for i in range(1, len(item)):
                        key2 += ','
                        key2 += item[i]  # columns[ord(item[i][0])-ord(A)]+item[i][1]
                    frequent_satisfaction[key2] = str(count) + " ; " + str(format(confidence2, '.00%'))  # format(count/total_count, '.00%')

file_object = open('F:\zhangyi\course\data mining\output\output_filter_sup_con_B4_5.txt', 'w')
for fre in frequent_satisfaction:
    file_object.write("{:<14}".format(fre)+": "+str(frequent_satisfaction[fre])+"\n")
file_object.close()

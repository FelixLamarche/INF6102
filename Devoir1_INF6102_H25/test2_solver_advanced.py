from utils import Node, Instance, Solution

from collections import defaultdict

import random
import math
import copy
import time

from itertools import groupby, combinations

modularityTime = 0
class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, neighbors, label=0):
        super().__init__(idx, neighbors)
        self.__label = label

    def set_label(self, new_label) :
        self.__label = new_label

    def label(self) :
        return self.__label
    
    def __repr__(self) -> str:
        return 'Node'+str(['idx:'+str(self.get_idx()),'label:'+str(self.__label),'neighbors:'+str(self.neighbors())])
    

def get_label_dict(instance):
    """
    Creates a dictionnary key : label, value : list of nodes with this label
    """
    labels_dict = defaultdict(list)
    for node in instance.nodes:
        labels_dict[node.label()].append(node)
    return labels_dict


def modularity(instance):
    """
    Computes and returns the modularity of the graph with the current communities
    """
    t0 = time.time()
    Q = 0
    M2 = 2 * instance.M
    label_dict = get_label_dict(instance)

    for node1 in instance.nodes:
        label1 = node1.label()
        for node2 in label_dict.get(label1): 
            P = node1.degree() * node2.degree() / M2
            Q += (1 - P) if node2.get_idx() in node1.neighbors() else -(P)
    global modularityTime
    modularityTime += time.time() - t0
    return Q / M2

def label_evaluation(instance, current_node, new_label, label_dict):
    """
    Computes and returns the term to optimize when updating a node's label
    """
    sum = 0
    for test_node in label_dict[new_label] :
        if test_node == current_node :
            continue
        P = current_node.degree() * test_node.degree() / (2*instance.M)
        sum += (1 - P) if test_node.get_idx() in current_node.neighbors() else -(P)
    return sum

def update_label(instance, current_node) :
    """
    Computes and returns the current node's label and the best new label
    """

    label_dict = get_label_dict(instance)
    possible_labels = [i.label() for i in instance.nodes if i.get_idx() in current_node.neighbors()]
    possible_labels = list(set(possible_labels))
    current_label = current_node.label()

    # Determine best label for the node
    best_score = label_evaluation(instance, current_node, current_label, label_dict)
    best_labels = [current_label]
    
    for label in possible_labels :
        score = label_evaluation(instance, current_node, label, label_dict)
        if score > best_score :
            best_score = score
            best_labels = [label]
        elif score == best_score :
            best_labels.append(label)
    new_label = random.choice(best_labels)

    if new_label != current_label :
        current_node.set_label(new_label)

    # # Update new label
    # if new_label != current_label : 
    #     #print(f'node {current_node}, new label : {new_label}')
    #     current_node.set_label(new_label)

def LPAm(instance, dict_nodes, nodes_to_test) :

    # Q = modularity(instance)
    # i = 0
    # step = 1

    # while i < len(nodes_to_test) :
    #     print(f'LPAM step {step}, nodes to test {len(nodes_to_test)}')
    #     for node in nodes_to_test :
    #         update_label(instance, node)
    #         new_Q = modularity(instance)
    #         if (math.isclose(new_Q, Q, rel_tol=1e-7)) :
    #             i += 1
    #         elif (new_Q > Q) :
    #             i = 0
    #         else :
    #             print("Erreur : la modularité a diminué")
    #             return
    #         Q = new_Q
            
    #     step += 1
    Q = modularity(instance)
    active_nodes = set(nodes_to_test)
    step = 1
    while active_nodes:
        print(f'LPAM step {step}, nodes to test {len(active_nodes)}')
        new_active_nodes = set()

        for node in active_nodes:
            old_label = node.label()
            update_label(instance, node)
            new_Q = modularity(instance)

            if new_Q > Q:
                # Add neighbors for potential updates
                new_active_nodes.update(dict_nodes[n] for n in node.neighbors())
            elif math.isclose(new_Q, Q, rel_tol=1e-7):
                continue  # Skip unchanged cases
            else:
                print("Error: modularity decreased")
                return

            Q = new_Q

        active_nodes = new_active_nodes
        step += 1

def merge_communities(instance, dict_nodes) :
    init_Q = modularity(instance)

    labels_dict = get_label_dict(instance)

    nodes_to_test = set()

    comb_Q_dict = {}

    # Get label duos
    label_duos = set()
    index_to_node = {node.get_idx(): node for nodes in labels_dict.values() for node in nodes}

    for label1, nodes in labels_dict.items():
        for node in nodes:
            for neighbor_idx in node.neighbors():
                neighbor = index_to_node.get(neighbor_idx)
                label2 = neighbor.label()
                if label1 != label2:
                    pair = tuple(sorted((label1, label2)))
                    label_duos.add(pair)

    # On teste les duos
    for duo in label_duos :
        original_labels = {i : i.label() for i in instance.nodes}

        # On merge deux communautés
        for node in instance.nodes :
            if node.label() == duo[0] :
                node.set_label(duo[1])

        # On regarde si le merge a amélioré Q
        duo_Q = modularity(instance)
        if duo_Q >= init_Q :
            comb_Q_dict[duo] = duo_Q

        # On reset pour continuer les tests
        for node in instance.nodes :
            node.set_label(original_labels[node])

    if comb_Q_dict == {} :
        return False, {}
    else :
        sorted_merges = sorted(comb_Q_dict.items(), key=lambda x: x[1], reverse=True)
        merged_labels = set()
        # On effectue les meilleurs merges
        for (label1, label2), _ in sorted_merges:
            if (label1 not in merged_labels) and (label2 not in merged_labels):
                for node in labels_dict[label1] :
                    node.set_label(label2)
                    for neighbor in node.neighbors() :
                        nodes_to_test.add(dict_nodes[neighbor])
                merged_labels.add(label1)
                merged_labels.add(label2)
                #print(f'Communities merged : {label1} {label2}')
        return True, nodes_to_test

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """
    for i, node in enumerate(instance.nodes):
        custom_node = CustomNode(node.get_idx(), node.neighbors(), label=node.get_idx())
        instance.nodes[i] = custom_node

    init_groups = [ [node] for node in instance.nodes]
    dict_nodes = {i.get_idx() : i for i in instance.nodes}
    
    total_step = 1
    merge = True
    nodes_to_test = instance.nodes

    while merge :
        print(total_step, modularity(instance))
        #print("début LPAM")
        LPAm(instance, dict_nodes, nodes_to_test)
        #print("fin LPAM, début merge")
        merge, nodes_to_test = merge_communities(instance, dict_nodes)
        #print("fin merge")
        total_step += 1

    sorted_nodes = sorted(instance.nodes, key=lambda node: node.label())
    groups = [list(map(lambda node: node.get_idx(), group)) for _, group in groupby(sorted_nodes, key=lambda node: node.label())]

    global modularityTime
    print(f"modularity time : {modularityTime}")

    return Solution(groups)
        

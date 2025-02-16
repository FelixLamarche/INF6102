from utils import Node, Instance, Solution

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, neighbors: list[int]):
        super().__init__(idx, neighbors)
        self.group_label = idx
    
    def is_adjacent_to(self, node: Node):
        return node.get_idx() in self.neighbors()
    
    def set_group_label(self, group_label: int):
        self.group_label = group_label

class Solver:
    def __init__(self, instance: Instance):
        for i, node in enumerate(instance.nodes):
            instance.nodes[i] = CustomNode(node.get_idx(), node.neighbors())
    
        self.instance = instance
        self.groups = {}
        self.__set_initial_groups()
        
        self.empty_group_labels = []
        self.groups_Q = {label: self.calculate_group_Q(label) for label, group in self.groups.items()}
        self.Q = sum(self.groups_Q.values())

    def to_solution(self) -> Solution:
        return Solution([ list(map(lambda node: node.get_idx(), group)) for group in self.groups.values() if len(group) > 0])

    def solve_LMPAm_and_greedy(self):
        prev_Q = self.Q
        MIN_DELTA_Q = 0.0001
        is_getting_better = True
        nb_iters = 1
        while is_getting_better:
            self.solve_LPAm()
            self.solve_greedy_merge()
            print("nb_iterations of LPAm + merge: " + str(nb_iters))
            print("Q: " + str(self.Q) + " prev_Q: " + str(prev_Q))
            nb_iters += 1

            if self.Q - prev_Q <= MIN_DELTA_Q:
                is_getting_better = False
            prev_Q = self.Q

    def solve_greedy_merge(self):
        MIN_DELTA_Q = 0.0001

        is_merging = True
        nb_iterations = 0
        while is_merging:
            prev_Q = self.Q
            for label_1, group_1 in self.groups.items():
                if len(group_1) == 0:
                    continue
                merging_label = -1
                max_delta_Q = 0
                max_merge_Q = 0

                # Cache the Q of the neighboring labels to avoid recalculating them
                group_merged_Qs = {}
                for node in group_1:
                    for neighbor_idx in node.neighbors():
                        label_2 = self.get_node(neighbor_idx).group_label
                        group_2 = self.groups[label_2]
                        if label_1 == label_2 or len(group_2) == 0 or label_2 in group_merged_Qs:
                            continue
                        merge_Q = self.calculate_Q_of_merge(label_1, label_2)
                        delta_merge_Q = self.calculate_Q_delta_of_merge(label_1, label_2, merge_Q)
                        group_merged_Qs[label_2] = merge_Q
                        if delta_merge_Q > max_delta_Q:
                            max_delta_Q = delta_merge_Q
                            max_merge_Q = merge_Q
                            merging_label = label_2
                if merging_label != -1:
                    self.merge_groups(label_1, merging_label, max_merge_Q)
                nb_iterations += 1
            self.__remove_empty_groups()

            print("Q: " + str(self.Q))
            print("nb_iterations: " + str(nb_iterations))
            # Halting condition
            if self.Q - prev_Q <= MIN_DELTA_Q:
                is_merging = False
    
    def solve_LPAm(self):
        MIN_DELTA_Q = 0.0001

        prev_Q = self.Q
        is_getting_better = True

        while is_getting_better:
            for node in self.instance.nodes:
                node_group = self.groups[node.group_label]
                node_group_Q_with = self.groups_Q[node.group_label]
                
                node_group_Q_without = self.calculate_group_Q_without_node(node.group_label, node)

                best_Q_delta = 0
                best_group_label = node.group_label
                best_group_new_Q = 0

                # Cache the Q of the neighboring labels to avoid recalculating them
                neighbors_Qs = {}
                for neighbor_idx in node.neighbors():
                    neighbor_label = self.get_node(neighbor_idx).group_label
                    if neighbor_label == node.group_label or neighbor_label in neighbors_Qs:
                        continue
                    neighbor_group = self.groups[neighbor_label]
                    neighbor_group_Q_without = self.groups_Q[neighbor_label]

                    neighbor_group_Q_with = self.calculate_group_Q_with_node(neighbor_label, node)
                    neighbors_Qs[neighbor_label] = neighbor_group_Q_with

                    Q_delta = neighbor_group_Q_with - neighbor_group_Q_without + node_group_Q_without - node_group_Q_with
                    if Q_delta > best_Q_delta:
                        best_Q_delta = Q_delta
                        best_group_label = neighbor_label
                        best_group_new_Q = neighbor_group_Q_with
                
                if best_group_label != node.group_label:
                    self.swap_node_group(node, node.group_label, best_group_label, node_group_Q_without, best_group_new_Q)
                self.__remove_empty_groups()

            # Halting condition
            print("Q: " + str(self.Q) + " prev_Q: " + str(prev_Q))
            if self.Q - prev_Q <= MIN_DELTA_Q:
                is_getting_better = False
            prev_Q = self.Q


    def swap_node_group(self, node: CustomNode, old_group_label: int, new_group_label: int, old_group_new_Q: float, new_group_new_Q: float):
        self.groups[old_group_label].remove(node)
        self.groups[new_group_label].append(node)
        node.set_group_label(new_group_label)
        
        # Then Recalculate Q for those two groups and update the global Q
        self.__update_Q_of_group(old_group_label, old_group_new_Q)
        self.__update_Q_of_group(new_group_label, new_group_new_Q)

    def merge_groups(self, group_1_label: int, group_2_label: int, merged_Q: float):
        group_1 = self.groups[group_1_label]
        group_2 = self.groups[group_2_label]
        for node in group_2:
            node.set_group_label(group_1_label)
        group_1.extend(group_2)
        self.groups[group_2_label] = []

        self.empty_group_labels.append(group_2_label)

        self.__update_Q_of_group(group_1_label, merged_Q)
        self.__update_Q_of_group(group_2_label, 0)

    def calculate_Q_delta_of_merge(self, group_1_label: int, group_2_label: int, merged_Q: float) -> float:
        return merged_Q - self.groups_Q[group_1_label] - self.groups_Q[group_2_label]

    def calculate_Q_of_merge(self, group_1_label: int, group_2_label: int) -> float:
        group_1 = self.groups[group_1_label]
        group_2 = self.groups[group_2_label]

        group_1_Q = self.groups_Q[group_1_label]
        group_2_Q = self.groups_Q[group_2_label]

        M2 = 2 * self.instance.M
        Q_delta = 0
        # Get the delta of the merge of the two groups
        for node1 in group_1:
            for node2 in group_2:
                P = node1.degree() * node2.degree() / M2
                Q_delta += (1 - P) if node2.is_adjacent_to(node1) else -(P)
        
        Q_delta *= 2
        Q_delta /= M2
        return group_1_Q + group_2_Q + Q_delta

    def calculate_group_Q(self, group_label: int) -> float:
        M2 = 2 * self.instance.M
        Q_group = 0
        group = self.groups[group_label]
        for node1 in group:
            for node2 in group:
                P = node1.degree() * node2.degree() / M2
                Q_group += (1 - P) if node2.is_adjacent_to(node1) else -(P)
        return Q_group / M2
    
    def calculate_group_Q_with_node(self, group_label: int, node: CustomNode) -> float:
        M2 = 2 * self.instance.M
        group = self.groups[group_label]
        Q_group = self.groups_Q[group_label] * M2
        Q_delta = 0
        # Add the delta of the node with all the other nodes in the group
        for node2 in group:
            P = node.degree() * node2.degree() / M2
            Q_delta += (1 - P) if node2.is_adjacent_to(node) else -(P)

        # Multiply by 2 for the symmetry of (nodeX with nodeY) and (nodeY with nodeX)
        Q_delta *= 2
        # add the delta of the node with itself
        P = node.degree() * node.degree() / M2
        Q_delta += -(P)
        return (Q_group + Q_delta) / M2

    def calculate_group_Q_without_node(self, group_label: int, node: CustomNode) -> float:
        M2 = 2 * self.instance.M
        group = self.groups[group_label]
        Q_group = self.groups_Q[group_label] * M2

        Q_delta = 0
        # Add the delta of the node with all the other nodes in the group
        for node2 in group:
            P = node.degree() * node2.degree() / M2
            Q_delta += (1 - P) if node2.is_adjacent_to(node) else -(P)

        # Multiply by 2 for the symmetry of (nodeX with nodeY) and (nodeY with nodeX)
        Q_delta *= 2
        # add the delta of the node with itself
        P = node.degree() * node.degree() / M2
        Q_group += -(P)

        return (Q_group - Q_delta) / M2
    
    def get_node(self, idx: int) -> CustomNode:
        return self.instance.nodes[idx - 1]

    def __remove_empty_groups(self):
        for group_label_to_remove in self.empty_group_labels:
            self.groups.pop(group_label_to_remove)
            self.groups_Q.pop(group_label_to_remove)
        self.empty_group_labels = []

    def __update_Q_of_group(self, group_label :int, new_group_Q: float):
        old_group_Q = self.groups_Q[group_label]
        self.groups_Q[group_label] = new_group_Q
        self.Q += new_group_Q - old_group_Q

    def __set_initial_groups(self):
        for i, node in enumerate(self.instance.nodes):
            node.set_group_label(i)
            self.groups[i] = [node]


def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """
    
    solver = Solver(instance)
    solver.solve_LMPAm_and_greedy()

    print("Q: " + str(solver.Q))
    sol = solver.to_solution()
    return sol

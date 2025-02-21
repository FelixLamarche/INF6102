from utils import Node, Instance, Solution

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, neighbors: list[int]):
        super().__init__(idx, neighbors)
        self.group_label = idx
        self.k = len(neighbors) # Degree of the node
        self.idx = idx
        self.neighbor_set = set(neighbors)
    
    def is_adjacent_to(self, node: Node) -> bool:
            return node.idx in self.neighbor_set
    
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

        self.group_Q_merges = {}

    def to_solution(self) -> Solution:
        return Solution([ list(map(lambda node: node.get_idx(), group)) for group in self.groups.values() if len(group) > 0])

    def solve_LMPAm_and_greedy(self):
        prev_Q = self.Q
        MIN_DELTA_Q = 0.0001
        is_getting_better = True
        nb_iters = 0
        while is_getting_better:
            print("\nLPAm iteration: " + str(nb_iters))
            self.solve_LPAm()
            print("\nMerge iteration: " + str(nb_iters))
            self.solve_greedy_merge()

            nb_iters += 1
            print("\nQ: " + str(self.Q) + " | nb_iterations of LPAm + merge: " + str(nb_iters) + "\n")

            if self.Q - prev_Q <= MIN_DELTA_Q:
                is_getting_better = False
            prev_Q = self.Q

    def solve_greedy_merge(self):
        MIN_DELTA_Q = 0.0001

        is_merging = True
        nb_iterations = 0
        nb_merges = 0
        while is_merging:
            prev_Q = self.Q
            for label_1, group_1 in self.groups.items():
                if len(group_1) == 0:
                    continue
                merging_label = -1
                max_delta_Q = 0
                max_merge_Q = 0
                group_1_Q = self.groups_Q[label_1]
                group_1_ID = str(label_1) + "-" + str(group_1_Q)

                if group_1_ID not in self.group_Q_merges:
                    self.group_Q_merges[group_1_ID] = {}

                for node in group_1:
                    for neighbor_idx in node.neighbors():
                        label_2 = self.get_node(neighbor_idx).group_label
                        group_2 = self.groups[label_2]
                        group_2_Q = self.groups_Q[label_2]
                        group_2_ID = str(label_2) + "-" + str(group_2_Q)

                        if label_1 == label_2 or len(group_2) == 0 or (group_2_ID in self.group_Q_merges[group_1_ID]):
                            continue
                        merge_Q = self.calculate_Q_of_merge(label_1, label_2)
                        delta_merge_Q = self.calculate_Q_delta_of_merge(label_1, label_2, merge_Q)

                        # Cache results
                        self.group_Q_merges[group_1_ID][group_2_ID] = merge_Q
                        if group_2_ID not in self.group_Q_merges:
                            self.group_Q_merges[group_2_ID] = {}
                        self.group_Q_merges[group_2_ID][group_1_ID] = merge_Q

                        if delta_merge_Q > max_delta_Q:
                            max_delta_Q = delta_merge_Q
                            max_merge_Q = merge_Q
                            merging_label = label_2
                if merging_label != -1:
                    self.merge_groups(label_1, merging_label, max_merge_Q)
                    nb_merges += 1

                nb_iterations += 1
            self.__remove_empty_groups()

            print("Q: " + str(self.Q) + " | nbMerges: " + str(nb_merges) + " | nbIterations: " + str(nb_iterations))
            # Halting condition
            if self.Q - prev_Q <= MIN_DELTA_Q:
                is_merging = False
    
    def solve_LPAm(self):
        MIN_DELTA_Q = 0.0001

        prev_Q = self.Q
        is_getting_better = True

        while is_getting_better:
            nb_swaps = 0
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
                    nb_swaps += 1
                self.__remove_empty_groups()

            # Halting condition
            print("Q: " + str(self.Q) + " | nbNodeSwaps: " + str(nb_swaps))
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

        # Check if swapping the node created disconnections
        if self.is_group_disconnected(old_group_label):
            new_labels = self.split_disconnected_group(old_group_label)
            for new_label in new_labels:
                self.__update_Q_of_group(new_label, self.calculate_group_Q(new_label))

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
                P = node1.k * node2.k / M2
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
                P = node1.k * node2.k / M2
                Q_group += (1 - P) if node2.is_adjacent_to(node1) else -(P)
        return Q_group / M2
    
    def calculate_group_Q_with_node(self, group_label: int, node: CustomNode) -> float:
        M2 = 2 * self.instance.M
        group = self.groups[group_label]
        Q_group = self.groups_Q[group_label] * M2
        Q_delta = 0
        # Add the delta of the node with all the other nodes in the group
        for node2 in group:
            P = node.k * node2.k / M2
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
            P = node.k * node2.k / M2
            Q_delta += (1 - P) if node2.is_adjacent_to(node) else -(P)

        # Multiply by 2 for the symmetry of (nodeX with nodeY) and (nodeY with nodeX)
        Q_delta *= 2
        # add the delta of the node with itself
        P = node.k * node.k / M2
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
        for i, node in sorted(enumerate(self.instance.nodes), key=lambda x: len(x[1].neighbors()), reverse=True):
            node.set_group_label(i)
            self.groups[i] = [node]

    def is_group_disconnected(self, group_label: int) -> bool:
        group = self.groups[group_label]
        if not group:
            return False
        visited = set()
        queue = [group[0]]  # Start from any node in the group
        while queue:
            node = queue.pop()
            if node.get_idx() in visited:
                continue
            visited.add(node.get_idx())
            for neighbor_idx in node.neighbors():
                neighbor = self.get_node(neighbor_idx)
                if neighbor.group_label == group_label and neighbor.get_idx() not in visited:
                    queue.append(neighbor)
        return len(visited) != len(group)
    
    def split_disconnected_group(self, group_label:int) -> list[int]:
        group = self.groups[group_label]
        visited = set()
        subgroups = []

        for node in group:
            if node.get_idx() not in visited:
                new_subgroup = []
                queue = [node]
                while queue:
                    current_node = queue.pop()
                    if current_node.get_idx() not in visited:
                        visited.add(current_node.get_idx())
                        new_subgroup.append(current_node)
                        for neighbor_idx in current_node.neighbors():
                            neighbor = self.get_node(neighbor_idx)
                            if neighbor.get_idx() not in visited and neighbor in group:
                                queue.append(neighbor)
                subgroups.append(new_subgroup)

        # Remove the original group and assign new labels
        self.groups.pop(group_label)
        new_labels = []
        for i, subgroup in enumerate(subgroups):
            new_label = max(self.groups.keys(), default=-1) + 1
            new_labels.append(new_label)
            self.groups[new_label] = subgroup
            for node in subgroup:
                node.set_group_label(new_label)
            self.groups_Q[new_label] = self.calculate_group_Q(new_label)

        return new_labels



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

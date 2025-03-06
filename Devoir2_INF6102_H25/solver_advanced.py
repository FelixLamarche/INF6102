# By:
# Felix Lamarche - 2077446
# Auriane Peter–Hemon - 2310513

from utils import Node, Instance, Solution, Edge
import math
import random
import time

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, revenue):
        super().__init__(idx, revenue)

    def __repr__(self) -> str:
        return 'Node'+str(['idx:'+str(self.idx()),'revenue:'+str(self.revenue()), 'dist_to_root:'+str(self.dist_to_root)])

class SolverAdvanced:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.instance.nodes = {idx: CustomNode(idx, node.revenue()) for idx, node in self.instance.nodes.items()}
        self.nodes : dict[int, CustomNode] = self.instance.nodes
        
        self.min_dists_to_root = {idx: 9999999 for idx, node in self.nodes.items()}  # Minimum distance to the root node
        self.min_dists_to_root[self.instance.get_root().idx()] = 0
        self.dists_to_root = {idx: 999999 for idx, node in self.nodes.items()} # Distance to the root node currently
        self.dists_to_root[self.instance.get_root().idx()] = 0
        self.node_edges = {}
        self.fill_node_edges()
        self.fill_nodes_min_dist_to_root()

        self.max_cost = self.instance.B
        self.max_distance = self.instance.H
        self.revenue_nodes = {node.idx(): node for node in self.nodes.values() if node.revenue() > 0 }

        self.edges_solution = set()
        self.best_edge_solution = set()
        self.best_revenue = 0
        self.nodes_solution = set()
        self.cost = 0
        self.revenue = 0


    def solve_edges(self):
        self.init_solution_empty()

        queue = [(0, self.instance.get_root())]
        while len(queue) > 0:
            distance, current_node = queue.pop(0)

            if distance > self.max_distance:
                continue
            
            # Get the best edges in profit
            best_edges = self.get_best_edges_neighboring_edge(current_node, self.cost, self.edges_solution)
            if len(best_edges) == 0:
                continue

            # Choose the best edge
            best_edges = list(best_edges)
            best_edges.sort(key=lambda edge: edge.cost())
            best_edge = random.choice(best_edges)

            if best_edge.cost() + self.cost > self.max_cost:
                continue
            
            # Update queue
            # Re-add the current node as we added an edge from it
            queue.append((distance, current_node))
            # Add the connected node to the queue
            connected_node = self.get_other_node_of_edge(best_edge, current_node)
            queue.append((distance + 1, connected_node))

            # Add the edge to the solution
            self.edges_solution.add(best_edge)
            self.nodes_solution.add(current_node)
            self.cost += best_edge.cost()
            self.revenue += self.get_edge_revenue(best_edge, current_node)

    def solve_profit_nodes_simulated_annealing(self):
        """
        Solves the problem by looking at profit nodes and adding all of the edges and nodes to reach them
            Simulated Annealing Version
        """
        self.init_solution_empty()
        temp = 10000000000000 # to choose through experimentation
        cooling_rate = 0.999 # to choose through experimentation

        nb_iterations_before_amelioration = 75
        cur_iteration = 0
        prev_revenue = -1

        while cur_iteration < nb_iterations_before_amelioration:
            do_remove_node = False
            chosen_revenue_node = None
            chosen_revenue_cost_ratio = 0
            chosen_revenue = 0
            chosen_cost = 0
            chosen_path = list()
            chosen_nodes = list()
            chosen_edges = list()

            revenue_nodes_idx = list(self.revenue_nodes.keys())
            random.shuffle(revenue_nodes_idx)
            for profit_node_idx in revenue_nodes_idx:
                profit_node = self.revenue_nodes[profit_node_idx]
                if self.min_dists_to_root[profit_node.idx()] > self.max_distance:
                    continue

                if profit_node in self.nodes_solution:
                    if profit_node == self.instance.get_root():
                        continue
                    if len(set(self.instance.get_neighbors(profit_node)).intersection(self.edges_solution)) > 1:
                        continue

                    nodes_to_remove, edges_to_remove = self.get_path_to_remove_node(profit_node)
                    if len(nodes_to_remove) == 0:
                        continue

                    chosen_revenue = -sum(node.revenue() for node in nodes_to_remove if node.idx() in self.revenue_nodes)
                    chosen_cost = -sum(edge.cost() for edge in edges_to_remove)
                    chosen_nodes = nodes_to_remove
                    chosen_edges = edges_to_remove
                    do_remove_node = True

                    # Annealing check as removing a node always degrades our current solution
                    if self.probability_annealing(temp, self.revenue + chosen_revenue, self.revenue):
                        chosen_revenue_node = profit_node
                        break
                else:
                    path = self.get_additional_path_to_node(profit_node)
                    node_in_solution = path[0][1] # The first element in the path is a node in the solution, the last is the profit node
                    new_nodes = set(node for dist, node, edge in path[1:])
                    new_edges = set(edge for dist, node, edge in path[0:-1])

                    additional_cost = sum(edge.cost() for edge in new_edges if edge)
                    if additional_cost + self.cost > self.max_cost:
                        continue

                    additional_dist_to_root = self.dists_to_root[node_in_solution.idx()] 
                    dist_to_solution = path[-1][0] # The last element in the path is the final profit node
                    if dist_to_solution + additional_dist_to_root > self.max_distance:
                        continue

                    additional_revenue = sum(node.revenue() for node in new_nodes if node.idx() in self.revenue_nodes)
                    revenue_cost_ratio = additional_revenue / additional_cost
                    if revenue_cost_ratio > chosen_revenue_cost_ratio:
                        chosen_revenue_node = profit_node
                        chosen_revenue = additional_revenue
                        chosen_cost = additional_cost
                        chosen_revenue_cost_ratio = revenue_cost_ratio
                        chosen_path = path
                        chosen_nodes = new_nodes
                        chosen_edges = new_edges
                        do_remove_node = False
                        break
            
            # Add node to the solution with its path
            if chosen_revenue_node is not None and not do_remove_node:
                # Update distances to root
                additional_dist_to_root = self.dists_to_root[chosen_path[0][1].idx()] # The first element in the path is a node in the solution
                for dist, node, edge in chosen_path:
                    self.dists_to_root[node.idx()] = min(self.dists_to_root[node.idx()], additional_dist_to_root + dist)
                self.edges_solution.update(chosen_edges)
                self.nodes_solution.update(chosen_nodes)

                self.cost += chosen_cost
                self.revenue += chosen_revenue
            elif do_remove_node and chosen_revenue_node is not None:
                # Remove the node from the solution
                self.edges_solution.difference_update(chosen_edges)
                for node in chosen_nodes:
                    self.dists_to_root[node.idx()] = 999999
                self.nodes_solution.difference_update(chosen_nodes)

                self.cost += chosen_cost
                self.revenue += chosen_revenue

            #self.instance.visualize_solution(Solution(self.edges_solution))
            # Continue while there are ameliorations
            temp = temp * cooling_rate
            cur_iteration += 1
            if self.revenue > prev_revenue:
                cur_iteration = 0
            prev_revenue = self.revenue
            if self.revenue > self.best_revenue:
                cur_iteration = 0
                self.best_edge_solution = self.edges_solution.copy()
                self.best_revenue = self.revenue
                print(self.revenue)
            
        print(self.revenue)
        print("temp:" + str(temp), "cur_iteration:" + str(cur_iteration))


    def solve_profit_nodes_hill_climb(self):
        """
        Solves the problem by looking at profit nodes and adding all of the edges and nodes to reach them
            Hill Climbing Version
        """
        self.init_solution_empty()

        prev_revenue = -1
        while self.revenue > prev_revenue:
            prev_revenue = self.revenue

            best_revenue_node = None
            best_revenue_cost_ratio = 0
            best_revenue = 0
            best_cost = 0
            best_path = list()
            best_nodes = list()
            best_edges = list()

            revenue_nodes_idx = list(self.revenue_nodes.keys())
            random.shuffle(revenue_nodes_idx)
            for profit_node_idx in revenue_nodes_idx:
                profit_node = self.revenue_nodes[profit_node_idx]
                if profit_node in self.nodes_solution or self.min_dists_to_root[profit_node.idx()] > self.max_distance:
                    continue

                path = self.get_additional_path_to_node(profit_node)
                node_in_solution = path[0][1] # The first element in the path is a node in the solution, the last is the profit node
                new_nodes = set(node for dist, node, edge in path[1:])
                new_edges = set(edge for dist, node, edge in path[0:-1])

                additional_cost = sum(edge.cost() for edge in new_edges if edge)
                if additional_cost + self.cost > self.max_cost:
                    continue

                additional_dist_to_root = self.dists_to_root[node_in_solution.idx()] 
                dist_to_solution = path[-1][0] # The last element in the path is the final profit node
                if dist_to_solution + additional_dist_to_root > self.max_distance:
                    continue

                additional_revenue = sum(node.revenue() for node in new_nodes if node.idx() in self.revenue_nodes)
                revenue_cost_ratio = additional_revenue / additional_cost
                if revenue_cost_ratio > best_revenue_cost_ratio:
                    best_revenue_node = profit_node
                    best_revenue = additional_revenue
                    best_cost = additional_cost
                    best_revenue_cost_ratio = revenue_cost_ratio
                    best_path = path
                    best_nodes = new_nodes
                    best_edges = new_edges
            
            # Add node to the solution with its path
            if best_revenue_node is not None:
                # Update distances to root
                additional_dist_to_root = self.dists_to_root[best_path[0][1].idx()] # The first element in the path is a node in the solution
                for dist, node, edge in best_path:
                    self.dists_to_root[node.idx()] = min(self.dists_to_root[node.idx()], additional_dist_to_root + dist)
                self.edges_solution.update(best_edges)
                self.nodes_solution.update(best_nodes)

                self.cost += best_cost
                self.revenue += best_revenue

        self.best_revenue = self.revenue
        self.best_edge_solution = self.edges_solution.copy()


    def get_additional_path_to_node(self, node: CustomNode) -> list[tuple[int, CustomNode, Edge]]:
        """Returns the additional edges and nodes to reach the node from the current solution through a DFS"""
        if node in self.nodes_solution:
            return list()
        
        queue = [(0, node, None)]
        visited_nodes = dict()
        connected_element = None # First node found in the solution
        while len(queue) > 0 and connected_element is None:
            cur_element = queue.pop(0)
            dist, current_node, prev_edge = cur_element
            if current_node.idx() in visited_nodes:
                continue

            visited_nodes[current_node.idx()] = (dist, current_node, prev_edge)

            for edge in self.instance.get_neighbors(current_node):
                other_node = self.get_other_node_of_edge(edge, current_node)
                if other_node in self.nodes_solution:
                    connected_element = (dist + 1, other_node, edge)
                    visited_nodes[other_node.idx()] = connected_element
                    break
                else:
                    queue.append((dist + 1, other_node, edge))
        
        # Reconstruct the path
        path = list()
        max_dist = connected_element[0]

        prev_dist, prev_node, prev_edge = connected_element
        #path.append((max_dist - connected_element[0], prev_node, prev_edge))
        while True:
            dist, node, edge = visited_nodes[prev_node.idx()]
            dist = max_dist - dist # The distance is reversed, to get the distance from the node in the solution and not the profit node
            path.append((dist, node, edge))

            prev_edge = edge
            if prev_edge is None:
                break
            prev_node = self.get_other_node_of_edge(prev_edge, prev_node)

        return path
    
    def get_path_to_remove_node(self, node: CustomNode) -> tuple[set[CustomNode], set[Edge]]:
        """Returns the path exclusive to the node(in the solution) that can be removed with the specific node"""
        if node is self.instance.get_root():
            return set(), set()
        
        queue = list()
        visited_nodes = set()
        visited_nodes.add(self.instance.get_root()) # The root always remains

        edges_to_remove = set()
        nodes_to_remove = set()

        nodes_to_remove.add(node) # Always remove the node to remove
        edges_to_remove.update(set(self.instance.get_neighbors(node)).intersection(self.edges_solution))
        queue.extend(self.get_other_node_of_edge(edge, node) for edge in edges_to_remove)

        while len(queue) > 0:
            node = queue.pop(0)
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            if node in nodes_to_remove:
                continue

            node_edges = set(self.instance.get_neighbors(node)).intersection(self.edges_solution)
            neighbor_nodes = set(self.get_other_node_of_edge(edge, node) for edge in node_edges)
            neighbor_nodes.intersection_update(self.nodes_solution)
            neighbor_nodes.difference_update(nodes_to_remove)
            if len(neighbor_nodes) <= 1:
                nodes_to_remove.add(node)
                edges_to_remove.update(node_edges)
                if len(neighbor_nodes) == 1:
                    queue.append(neighbor_nodes.pop())

        return nodes_to_remove, edges_to_remove

    def probability_annealing(self, temp: float, new_revenue: int, cur_revenue: int) -> bool:
        if new_revenue > cur_revenue:
            return True
        f = math.exp((new_revenue - cur_revenue) / temp)
        #print(f, temp, new_revenue - cur_revenue)
        return random.random() < f

    def get_other_node_of_edge(self, edge: Edge, connected_node: CustomNode) -> CustomNode:
        return list(edge.idx().difference((connected_node,)))[0]

    def get_best_edges_neighboring_edge(self, node: CustomNode, cur_cost: int, edges_used: set[Edge]) -> set[Edge]:
        best_edges = set()
        best_edge_profit = 0

        for edge in self.instance.get_neighbors(node):
            if edge in edges_used:
                continue

            # If the edge is already in the solution
            if len(edge.idx().intersection(self.nodes_solution)) == 2:
                continue

            if edge.cost() + cur_cost <= self.max_cost:
                edge_profit = self.get_edge_revenue(edge, node)
                if edge_profit > best_edge_profit:
                    best_edges = set()
                    best_edges.add(edge)
                    best_edge_profit = edge_profit
                elif edge_profit == best_edge_profit:
                    best_edges.add(edge)

        return best_edges

    def get_edge_revenue(self, edge: Edge, connected_node: CustomNode) -> int:
        """Returns the added profit of the edge with the connected_node already in the solution"""
        other_idx = self.get_other_node_of_edge(edge, connected_node).idx()
        return self.revenue_nodes[other_idx].revenue() if other_idx in self.revenue_nodes else 0
    
    def fill_nodes_min_dist_to_root(self):
        queue = [(0, self.instance.get_root())]

        while len(queue) > 0:
            distance, current = queue.pop(0)
            self.min_dists_to_root[current.idx()] = min(distance, self.min_dists_to_root[current.idx()])

            for edge in self.instance.get_neighbors(current):
                other_node_idx = list(edge.idx().difference((current,)))[0].idx()
                if self.min_dists_to_root[other_node_idx] > distance + 1:
                    other_node = self.nodes[other_node_idx]
                    queue.append((distance + 1, other_node))

    def fill_node_edges(self):
        for node in self.nodes.values():
            self.node_edges[node.idx()] = set(edge for edge in self.instance.get_neighbors(node))

    def init_solution_empty(self):
        self.edges_solution = set()
        self.nodes_solution = set()
        self.nodes_solution.add(self.instance.get_root())
        self.revenue += self.instance.get_root().revenue()
        self.cost = 0
    
    def init_solution_full(self):
        self.edges_solution = set(edge for edge in self.instance.edges)
        self.cost = sum(edge.cost() for edge in self.edges_solution)

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """

    best_solution = None
    best_revenue = -1
    time_before = time.time()
    search_time = 10

    while time_before + search_time > time.time():
        solver = SolverAdvanced(instance)
        solver.solve_profit_nodes_simulated_annealing()

        if solver.best_revenue > best_revenue:
            best_solution = solver.best_edge_solution
            best_revenue = solver.best_revenue

    return Solution(best_solution)

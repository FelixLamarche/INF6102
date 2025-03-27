# By:
# Felix Lamarche - 2077446
# Auriane Peter–Hemon - 2310513

from utils import Node, Instance, Solution, Edge
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
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
    def __init__(self, instance: Instance, time_limit_sec: float = 290, best_edges : set[Edge] = set(), best_nodes: set[Node] = set()):
        self.time_before = time.time()
        self.time_limit = self.time_before + time_limit_sec

        self.instance = instance
        self.instance.nodes = {idx: Node(idx, node.revenue()) for idx, node in self.instance.nodes.items()}
        self.nodes : dict[int, Node] = self.instance.nodes
        
        self.min_dists_to_root = {} # Minimum distance to the root node if all edges are used
        self.node_edges = {} # Edges connected to each node
        self.fill_node_edges()
        self.fill_nodes_min_dist_to_root()

        self.max_cost = self.instance.B
        self.max_distance = self.instance.H
        self.revenue_nodes = {node.idx(): node for node in self.nodes.values() if node.revenue() > 0 }

        # Init edges and nodes of solution
        self.edges_solution = best_edges.copy()
        self.nodes_solution = best_nodes.copy()
        self.best_edge_solution = best_edges.copy()
        self.best_node_solution = best_nodes.copy()
        if self.instance.get_root() not in self.nodes_solution:
            self.nodes_solution.add(self.instance.get_root())
            self.best_node_solution.add(self.instance.get_root())
        self.cost = sum(edge.cost() for edge in self.edges_solution)
        self.revenue = sum(node.revenue() for node in self.nodes_solution if node.idx() in self.revenue_nodes)
        self.best_revenue = self.revenue

        self.dists_to_root = {} # Distance to the root node in the current solution
        self.fill_nodes_dist_to_root()


    def solve_profit_nodes_simulated_annealing(self, temp: int = 10000, cooling_rate: float = 0.99, tabu_iteration_length: int = 4):
        """
        Solves the problem by trying to add and remove profit nodes by adding all the missing node and edges to reach them
        Tries to pick the profit node which adds the most revenue, or can randomly add a or remove a profit node
            Simulated Annealing Version
            Gets around 7.5/8.0 on the test instances, and is slow
        """
        temp = self.instance.N * 250

        # Remove a random amount of profit nodes to get a more diverse starting solution
        profit_nodes = list(self.nodes_solution.intersection(self.revenue_nodes.values()))
        if len(profit_nodes) > 1:
            nb_profit_node_to_remove = random.randint(0, len(profit_nodes) - 1) # -1 to keep root node
            self.remove_amount_of_profit_nodes(nb_profit_node_to_remove)

        #self.visualize_solution(Solution(self.edges_solution), 0)

        nb_iterations_before_amelioration = 1500
        nb_iterations_before_change = 100
        cur_iteration_amelioration = 0
        cur_iteration_change = 0
        nb_iteration = 0
        prev_revenue = -1

        tabu_dict = dict()

        while self.has_time_left() and (cur_iteration_amelioration < nb_iterations_before_amelioration and cur_iteration_change < nb_iterations_before_change):
            do_remove_node = False
            chosen_revenue_node = None
            chosen_revenue = 0
            chosen_cost = 0
            chosen_path = list()
            chosen_nodes = list()
            chosen_edges = list()

            revenue_nodes_idx = list(self.revenue_nodes.keys())
            random.shuffle(revenue_nodes_idx)
            for profit_node_idx in revenue_nodes_idx:
                profit_node = self.revenue_nodes[profit_node_idx]

                # If the profit node is too far from the root
                if self.min_dists_to_root[profit_node.idx()] > self.max_distance:
                    continue
                if profit_node == self.instance.get_root():
                    continue
                # If the profit node is in the tabu list
                if profit_node.idx() in tabu_dict and tabu_dict[profit_node.idx()] > 0:
                    continue

                if profit_node in self.nodes_solution: # Check to remove the node
                    # If the the node has more than one edge in the solution, we don't try to remove it (as it would remove more than one profit node at a time)
                    if len(self.node_edges[profit_node.idx()].intersection(self.edges_solution)) > 1:
                        continue

                    nodes_to_remove, edges_to_remove = self.get_path_to_remove_node(profit_node)
                    if len(nodes_to_remove) == 0:
                        continue

                    difference_revenue = -sum(node.revenue() for node in nodes_to_remove if node.idx() in self.revenue_nodes)
                    # Annealing check as removing a node always degrades our current solution
                    if self.probability_annealing(temp, self.revenue + difference_revenue, self.revenue):
                        chosen_revenue_node = profit_node
                        chosen_revenue = difference_revenue
                        chosen_cost = -sum(edge.cost() for edge in edges_to_remove)
                        chosen_nodes = nodes_to_remove
                        chosen_edges = edges_to_remove
                        do_remove_node = True
                        # Stop searching after finding a node to remove
                        break
                else: # Node is not already in the solution
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
                    if additional_revenue > chosen_revenue:
                        chosen_revenue_node = profit_node
                        chosen_revenue = additional_revenue
                        chosen_cost = additional_cost
                        chosen_path = path
                        chosen_nodes = new_nodes
                        chosen_edges = new_edges
                        do_remove_node = False
                        # If a neighbor is better than the current solution, we add it and stop searching
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
                # Remove the nodes from the solution
                self.edges_solution.difference_update(chosen_edges)
                for node in chosen_nodes:
                    self.dists_to_root[node.idx()] = 999999
                self.nodes_solution.difference_update(chosen_nodes)

                self.cost += chosen_cost
                self.revenue += chosen_revenue

            # Update tabu list
            for node_idx in tabu_dict.keys():
                tabu_dict[node_idx] -= 1

            cur_iteration_amelioration += 1
            cur_iteration_change += 1
            nb_iteration += 1

            # Only cool down if we have changed the solution
            if chosen_revenue_node is not None:
                cur_iteration_change = 0
                temp = temp * cooling_rate
                tabu_dict[chosen_revenue_node.idx()] = tabu_iteration_length

            prev_revenue = self.revenue
            if self.revenue > self.best_revenue:
                print("Iteration: ", nb_iteration, "Revenue: ", self.revenue, "Cost: ", self.cost, "Temp: ", temp, "Time: ", time.time() - self.time_before)
                cur_iteration_amelioration = 0
                self.best_edge_solution = self.edges_solution.copy()
                self.best_node_solution = self.nodes_solution.copy()
                self.best_revenue = self.revenue

        print("STOP Iteration: ", nb_iteration, "Best Revenue: ", self.best_revenue, "Revenue: ", self.revenue, "Cost: ", self.cost, "Temp: ", temp, "Time: ", time.time() - self.time_before)
        print("\n")
            
    def solve_profit_nodes_hill_climb(self):
        """
        Solves the problem by looking at profit nodes and adding all of the edges and nodes to reach them
        Each iteration, adds the profit node (with all the edges and nodes to reach them) that has the best revenue/cost ratio
            Hill Climbing Version
            Gets around 7.2/8.0 on the test instances, and is quick
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
        self.best_node_solution = self.nodes_solution.copy()

    def destroy_and_reconstruct_edges(self):
        """Destroys the current solution by removing all the edges
        DOES NOT WORK PROPERLY"""

        nodes = list(self.nodes_solution)
        random.shuffle(nodes)

        self.edges_solution = set()
        self.nodes_solution = set()
        self.nodes_solution.add(self.instance.get_root())
        self.revenue = self.instance.get_root().revenue()
        self.cost = 0

        for node in nodes:
            path = self.get_additional_path_to_node(node)
            self.edges_solution.update(edge for dist, node, edge in path[0:-1])
            self.nodes_solution.update(node for dist, node, edge in path[1:])
            self.cost += sum(edge.cost() for dist, node, edge in path[0:-1])
            self.revenue += sum(node.revenue() for dist, node, edge in path[1:] if node.idx() in self.revenue_nodes)

    def remove_amount_of_profit_nodes(self, nb_profit_nodes_to_remove: int):
        MAX_ITERATION = 250
        
        profit_nodes = list(self.nodes_solution.intersection(self.revenue_nodes.values()))
        profit_nodes.remove(self.instance.get_root())
        nb_removed = 0
        nb_iteration = 0
        
        while nb_removed < nb_profit_nodes_to_remove and len(profit_nodes) > 0 and nb_iteration < MAX_ITERATION:
            nb_iteration += 1
            profit_node = random.choice(profit_nodes)
            # If the the node has more than one edge in the solution, we don't try to remove it (as it would remove more than one profit node at a time)
            if profit_node in self.nodes_solution and len(self.node_edges[profit_node.idx()].intersection(self.edges_solution)) == 1:
                nodes_to_remove, edges_to_remove = self.get_path_to_remove_node(profit_node)
                if len(nodes_to_remove) > 0:
                    self.edges_solution.difference_update(edges_to_remove)
                    for node in nodes_to_remove:
                        self.dists_to_root[node.idx()] = 999999
                    self.nodes_solution.difference_update(nodes_to_remove)

                    self.cost -= sum(edge.cost() for edge in edges_to_remove)
                    self.revenue -= sum(node.revenue() for node in nodes_to_remove if node.idx() in self.revenue_nodes)
                    nb_removed += 1
                    profit_nodes.remove(profit_node)

    def get_additional_path_to_node(self, node: Node) -> list[tuple[int, Node, Edge]]:
        """Returns the additional edges and nodes that are not in the solution to reach the given node from the current solution through a DFS
        The first node in the path is a node in the solution, the last is the profit node"""
        # To test: try to return the path with the lowest cost and not only the lowest distance
        if node in self.nodes_solution:
            return list()
        
        # Dist, Cost, Node, Edge
        queue = [(0, 0, node, None)]
        visited_nodes = dict()
        connected_element = None # First node found in the solution
        while len(queue) > 0:
            cur_element = queue.pop(0)
            dist, cur_cost, current_node, prev_edge = cur_element
            if current_node.idx() in visited_nodes and visited_nodes[current_node.idx()][1] <= cur_cost:
                continue
                
            visited_nodes[current_node.idx()] = (dist, cur_cost, current_node, prev_edge)

            for edge in self.node_edges[current_node.idx()]:
                other_node = self.get_other_node_of_edge(edge, current_node)
                new_element = (dist + 1, edge.cost() + cur_cost, other_node, edge)
                if other_node in self.nodes_solution:
                    if connected_element == None or (new_element[0] + self.dists_to_root[other_node.idx()] < self.max_distance and new_element[1] < connected_element[1]):
                        connected_element = new_element
                        visited_nodes[other_node.idx()] = new_element
                # If too slow, stop queueing nodes that are too far
                elif new_element[0] < self.max_distance and (connected_element == None or new_element[1] < connected_element[1]):
                    queue.append(new_element)
        
        # Reconstruct the path
        path = list()
        max_dist = connected_element[0]
        prev_dist, prev_cost, prev_node, prev_edge = connected_element
        while True:
            dist, cost, node, edge = visited_nodes[prev_node.idx()]
            dist = max_dist - dist # The distance is reversed, to get the distance from the node in the solution and not the profit node
            path.append((dist, node, edge))

            prev_edge = edge
            if prev_edge is None:
                break
            prev_node = self.get_other_node_of_edge(prev_edge, prev_node)

        return path

    def get_path_to_remove_node(self, node: Node) -> tuple[set[Node], set[Edge]]:
        """Returns the path exclusive to the node(in the solution) that can be removed including the node itself
        Works well only if the node has degree 1
        """
        if node is self.instance.get_root():
            return set(), set()
        
        queue = list()
        visited_nodes = set()
        visited_nodes.add(self.instance.get_root()) # The root always remains

        edges_to_remove = set()
        nodes_to_remove = set()

        nodes_to_remove.add(node) # Always remove the node to remove
        edges_to_remove.update(self.node_edges[node.idx()].intersection(self.edges_solution))
        queue.extend(self.get_other_node_of_edge(edge, node) for edge in edges_to_remove)

        while len(queue) > 0:
            cur_node = queue.pop(0)
            if cur_node in visited_nodes:
                continue
            visited_nodes.add(cur_node)
            if cur_node in nodes_to_remove:
                continue
            if cur_node.idx() in self.revenue_nodes and cur_node != node:
                continue

            node_edges = self.node_edges[cur_node.idx()].intersection(self.edges_solution)
            neighbor_nodes = set(self.get_other_node_of_edge(edge, cur_node) for edge in node_edges)
            neighbor_nodes.intersection_update(self.nodes_solution)
            neighbor_nodes.difference_update(nodes_to_remove)
            if len(neighbor_nodes) <= 1:
                nodes_to_remove.add(cur_node)
                edges_to_remove.update(node_edges)
                if len(neighbor_nodes) == 1:
                    queue.append(neighbor_nodes.pop())

        return nodes_to_remove, edges_to_remove

    def get_path_to_remove_node2(self, node: Node) -> tuple[set[Node], set[Edge]]:
        """
        Returns all the nodes and edges that should be removed to remove the target node from the solution
        Including the nodes that are disconnected from the root node, and the edges that are disconnected from the solution
        and the nodes that are with no profits that that aren't necessary to reahch any profit node
        N.B. Is slower than the other method
        """
        removed_node_edges = set()
        for edge in self.node_edges[node.idx()]:
            if edge in self.edges_solution:
                removed_node_edges.add(edge)

        # Remove the node to ease the search
        nodes_to_remove, edges_to_remove = self.get_path_to_remove_node(node)

        self.nodes_solution.difference_update(nodes_to_remove)
        self.edges_solution.difference_update(edges_to_remove)

        kept_edges = set()
        kept_nodes = set()

        queue = [self.instance.get_root()]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            kept_nodes.add(cur_node)
            for edge in self.node_edges[cur_node.idx()]:
                if edge in self.edges_solution:
                    kept_edges.add(edge)
                    other_node = self.get_other_node_of_edge(edge, cur_node)
                    if other_node not in kept_nodes:
                        queue.append(other_node)

        # Re-add the node to not modify the solution
        self.nodes_solution.update(nodes_to_remove)
        self.edges_solution.update(edges_to_remove)

        to_remove_nodes = self.nodes_solution.difference(kept_nodes)
        to_remove_edges = self.edges_solution.difference(kept_edges)

        return to_remove_nodes, to_remove_edges

    def probability_annealing(self, temp: float, new_revenue: int, cur_revenue: int) -> bool:
        """Returns True if we should accept the new revenue according to the simulated annealing probability"""
        if new_revenue > cur_revenue:
            return True
        f = math.exp((new_revenue - cur_revenue) / temp)
        return random.random() < f

    def get_other_node_of_edge(self, edge: Edge, connected_node: Node) -> Node:
        return list(edge.idx().difference((connected_node,)))[0]
    
    def fill_nodes_dist_to_root(self):
        """Fills the distance to the root node for each node in the solution"""
        self.dists_to_root = {idx: 9999999 for idx, node in self.nodes.items()} 
        self.dists_to_root[self.instance.get_root().idx()] = 0 # The root node is at distance 0

        queue = [(0, self.instance.get_root())]

        while len(queue) > 0:
            distance, current = queue.pop(0)
            self.dists_to_root[current.idx()] = min(distance, self.dists_to_root[current.idx()])

            for edge in set(self.node_edges[current.idx()]).intersection(self.edges_solution):
                other_node_idx = self.get_other_node_of_edge(edge, current).idx()
                if self.dists_to_root[other_node_idx] > distance + 1:
                    other_node = self.nodes[other_node_idx]
                    queue.append((distance + 1, other_node))

    def fill_nodes_min_dist_to_root(self):
        """Fills the minimum possible distance to the root node for each node if all edges are used"""
        if len(self.min_dists_to_root) > 0:
            return

        self.min_dists_to_root = {idx: 9999999 for idx, node in self.nodes.items()}  
        self.min_dists_to_root[self.instance.get_root().idx()] = 0 # The root node is at distance 0

        queue = [(0, self.instance.get_root())]

        while len(queue) > 0:
            distance, current = queue.pop(0)
            self.min_dists_to_root[current.idx()] = min(distance, self.min_dists_to_root[current.idx()])

            for edge in self.node_edges[current.idx()]:
                other_node_idx = list(edge.idx().difference((current,)))[0].idx()
                if self.min_dists_to_root[other_node_idx] > distance + 1:
                    other_node = self.nodes[other_node_idx]
                    queue.append((distance + 1, other_node))

    def fill_node_edges(self):
        """Fills the node_edges dictionary with the edges connected to each node"""
        if len(self.node_edges) > 0:
            return
        for node in self.nodes.values():
            self.node_edges[node.idx()] = set(edge for edge in self.instance.get_neighbors(node))

    def init_solution_empty(self):
        """Initializes the solution with no edges and only the root node"""
        self.edges_solution = set()
        self.nodes_solution = set()
        self.nodes_solution.add(self.instance.get_root())
        self.revenue = sum(node.revenue() for node in self.nodes_solution if node.idx() in self.revenue_nodes)
        self.cost = 0

        self.fill_nodes_dist_to_root()
    
    def init_solution_full(self):
        """Initializes the solution with all edges"""
        self.edges_solution = set(edge for edge in self.instance.edges)
        self.cost = sum(edge.cost() for edge in self.edges_solution)

    def has_time_left(self) -> bool:
        return time.time() < self.time_limit

    ######################
    ### UTILITIES
    ######################

    def visualize_solution(self, sol: Solution, iter: int):
        """
            Show and save the solution's visualization
        """
        figure_size = (18,14) 

        G = nx.Graph()
        G.add_nodes_from([n for n in self.instance.nodes.values()])
        G.add_edges_from([e.idx() for e in self.instance.edges])
        pos = nx.bfs_layout(G, self.instance.get_root(), align="horizontal")
        pos[self.instance.get_root()] = (0,0)
        if len(self.instance.edges) >= 1000:
            k = 15/np.sqrt(len(G.nodes()))
        else:
            k = 6/np.sqrt(len(G.nodes()))
        pos = nx.spring_layout(G, k=k, pos=pos, fixed=[self.instance.get_root()], seed=38206, scale=10)
        # Nodes colored by cluster
        fig, _ = plt.subplots(figsize=figure_size)

        # nx.draw(G, pos=pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, [self.instance.get_root()], node_shape="s", node_color="red", node_size=750)
        nx.draw_networkx_nodes(G, pos, set(self.instance.profit_nodes.values()).difference((self.instance.get_root(),)), node_shape="v", node_color="orange", node_size=500)
        nx.draw_networkx_nodes(G, pos, {n for n in self.instance.nodes.values()}.difference(self.instance.profit_nodes.values()),)
        
        path = list(map(lambda x: tuple(x.idx()),sol.get_path()))
        nx.draw_networkx_edges(G, pos, path)

        node_labels: dict[Node, str] = {}
        edge_labels: dict[tuple[Node,Node], str] = {}
        for edge in sol.get_path():
            edge_labels[tuple(edge.idx())] = edge.cost()
            for node in edge.idx():
                if node.revenue() > 0:
                    node_labels[node] = str(node.revenue())
                else:
                    node_labels[node] = ""

        nx.draw_networkx_labels(G, pos, node_labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)


        multi_path = self.instance.break_path(sol.get_path())
        colors = self.instance.generate_distinct_colors(len(multi_path))
        for i_path, color in zip(multi_path, colors): 
            nx.draw_networkx_edges(G, pos, i_path, edge_color=color, width=15, alpha=0.5)

        revenu = self.instance.solution_value(sol)
        consummed = self.instance.solution_cost(sol)
        fig.suptitle(f"Solution de {self.instance.filepath.stem}\nBudget consommé = {consummed}, Revenu = {revenu}", fontsize=18)
        fig.tight_layout()
        plt.savefig("visualization_"+str(iter)+".png")


    ######################
    ### OLD FUNCTIONS
    ######################

    def solve_edges(self):
        """
        Solves the problem by adding the best edges (that add the most profit) to the solution
        N.B. Does not work well
        """
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

    def get_best_edges_neighboring_edge(self, node: Node, cur_cost: int, edges_used: set[Edge]) -> set[Edge]:
        """
        Returns the edges connected to the node that are not in the solution and that can be added to the solution
        which increases the profit the most
        N.B. Does not work properly
        """
        best_edges = set()
        best_edge_profit = 0

        for edge in self.node_edges[node.idx()]:
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
    
    def get_edge_revenue(self, edge: Edge, connected_node: Node) -> int:
        """Returns the added profit of the edge with the connected_node already in the solution"""
        other_idx = self.get_other_node_of_edge(edge, connected_node).idx()
        return self.revenue_nodes[other_idx].revenue() if other_idx in self.revenue_nodes else 0
    

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """

    ### TO MODIFY
    SEARCH_TIME_SEC = 60
    ###
    TIME_MARGIN_SEC = 5

    best_edge_solution = None
    best_node_solution = None
    best_revenue = -1

    time_before = time.time()
    time_limit = time_before + SEARCH_TIME_SEC - TIME_MARGIN_SEC
    while time.time() < time_limit:
        time_left = time_limit - time.time()
        if best_edge_solution is None:
            solver = SolverAdvanced(instance, time_left)
            solver.solve_profit_nodes_hill_climb() 
            #solver.solve_profit_nodes_simulated_annealing()
        else:
            solver = SolverAdvanced(instance, time_left, best_edge_solution, best_node_solution)
            solver.solve_profit_nodes_simulated_annealing()
        if solver.best_revenue > best_revenue:
            best_edge_solution = solver.best_edge_solution.copy()
            best_node_solution = solver.best_node_solution.copy()
            best_revenue = solver.best_revenue

    return Solution(best_edge_solution)

def solve_find_parameter(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """

    best_solution = None
    best_revenue = -1
    time_before = time.time()
    search_time = 1

    results = []
    best_results = {}

    temps = [100 * (2 ** i) for i in range(20)]
    cooling_rates = [0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]
    tabu_iteration_lengths = [2, 4, 8, 16, 32, 64]


    for temp in temps:
        for cooling_rate in cooling_rates:
            for tabu_iteration_length in tabu_iteration_lengths:
                time_before = time.time()
                best_solver = None
                while time_before + search_time > time.time():
                    solver = SolverAdvanced(instance)
                    solver.solve_profit_nodes_simulated_annealing(temp, cooling_rate, tabu_iteration_length)
                    if solver.best_revenue > best_revenue:
                        best_solution = solver.best_edge_solution
                        best_revenue = solver.best_revenue
                        best_solver = solver

                result = (instance.filepath.name, solver.best_revenue, temp, cooling_rate, tabu_iteration_length, solver.max_cost, time.time() - time_before)
                results.append(result)

                if instance.filepath.name not in best_results or best_results[instance.filepath.name][1] < solver.best_revenue:
                    best_results[instance.filepath.name] = result

                print(result[0], "Revenue: " + str(result[1]), "Temp: " + str(result[2]), "cooling_rate: " + str(result[3]), "Tabu_Length: " + str(result[4]), "Time: " + str(result[6]))


    #for result in results:
        #print(result[0], "Revenue: " + str(result[1]), "Temp: " + str(result[2]), "cooling_rate: " + str(result[3]), "Tabu_Length: " + str(result[4]))

    print("BEST RESULTS")
    print("***********************************************************")
    print(best_results)

    with open('results'+instance.filepath.name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for result in sorted(results, key=lambda x: x[1], reverse=True):
            writer.writerow([result[0], "Revenue: " + str(result[1]), "Temp: " + str(result[2]), "cooling_rate: " + str(result[3]), "Tabu_Length: " + str(result[4]), "Time: " + str(result[6])])

    return Solution(best_solution)
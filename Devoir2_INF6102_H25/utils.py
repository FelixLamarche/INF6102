import os
import distinctipy
import pickle
import heapq


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import Iterable
from collections import deque
from pathlib import Path

"""
    Global visualization variables.
    Modify to your own convinience
"""
figure_size = (18,14) 

def make_universal(filepath):
    """
        Create an exploitable path for every operating systems
    """
    return os.sep.join(filepath.split('/'))

class Node:
    def __init__(self,idx: int, revenue: int=0):
        self.__idx = idx
        self.__revenue = revenue

    def idx(self):
        return self.__idx
    
    def revenue(self) -> int:
        return self.__revenue

    def __hash__(self) -> int:
        return self.__idx
    
    def __eq__(self, other):
        return self.__idx == other.__idx
    
    def __lt__(self, other):
        return self.__idx < other.__idx

    def __repr__(self) -> str:
        return 'Node'+str(['idx:'+str(self.__idx),'revenue:'+str(self.__revenue)])
    
class Edge:
    def __init__(self,n1: Node,n2: Node, cost=0):
        self.__idx = frozenset((n1,n2))
        self.__cost = cost
        ## Added
        self.__previous_cost = 0
    
    def idx(self):
        return self.__idx
    
    def cost(self):
        return self.__cost
    
    def has_node(self,u:Node | int) -> bool:
        if isinstance(u, Node):
            return u in self.__idx
        else:
            return Node(u) in self.__idx

    def __hash__(self):
        return self.__idx.__hash__()
    
    def __eq__(self, other:Node):
        return self.__idx == other.__idx

    def __repr__(self):
        return 'Edge'+str(['idx:'+str(tuple(self.__idx)),'cost:'+str(self.__cost)])
    
    # Added
    def set_new_cost(self, new_cost) :
        self.__previous_cost = self.__cost
        self.__cost = new_cost

    def set_back_old_cost(self) :
        self.__cost = self.__previous_cost

class Solution:
    def __init__(self, path: Iterable[Edge]):
        self.__path = list(path)
        self.T = len(self.__path)

    def get_path(self) -> list[Edge]:
        return self.__path

    def __repr__(self):
        return 'Solution'+str(self.__path)
    
class Instance:
    def __init__(self,in_file: str):
        self.filepath = Path(make_universal(in_file))
        assert(self.filepath.exists())
        with open(self.filepath) as f:
            lines = list([[x.strip() for x in x.split(' ') if x.strip() != ''] for x in f.readlines()])
            self.N, self.M, self.P, self.B, self.H = tuple(map(int,lines[0]))
            self.profit_nodes = {int(line[1]): Node(int(line[1]), int(line[2])) for line in filter(lambda line : line[0] == "PV", lines[1:])}
            self.nodes: dict[int, Node] = pickle.loads(pickle.dumps(self.profit_nodes))
            self.edges = set()

            for line in filter(lambda line : line[0] == "E", lines[1:]):
                n1 = int(line[1])
                n2 = int(line[2])
                cost = int(line[3])
                if n1 not in self.nodes.keys():
                    self.nodes[n1] = Node(n1)
                if n2 not in self.nodes.keys():
                    self.nodes[n2] = Node(n2)
                self.edges.add(Edge(self.nodes[n1], self.nodes[n2], cost))

        assert(self.N == len(self.nodes))
        assert(self.P == len(self.profit_nodes))
        assert(self.M == len(self.edges))

    def get_root(self) -> Node:
        """
            Return the root (distribution center) node
        """
        assert(1 in self.nodes.keys())
        return self.nodes[1]

    def get_node(self, idx: int) -> Node:
        """
            Return node from its id
        """
        assert(idx > 0)
        assert(idx <= self.N)
        return self.nodes[idx]
    
    def get_neighbors(self, node: Node) -> Iterable[Edge]:
        """
            Return all node neighbors
        """
        for edge in filter(lambda e: node in e.idx(), self.edges):
            yield edge

    def is_valid_solution(self,sol:Solution) -> bool:
        """ 
            Returns True when the solution is valid
        """
        return self.no_added_edges(sol) and self.all_edge_linked_to_root(sol) \
            and self.all_node_within_maximal_distance(sol) and self.is_budget_not_exceeded(sol)

    def no_added_edges(self,sol:Solution) -> bool:
        """
            Returns True when the solution has no added edges
        """
        return len(self.added_edges(sol))==0 

    def added_edges(self,sol:Solution)-> set[Edge]:
        """
            Returns all nodes missing from the group assignement
        """
        edge_set = {edge for edge in self.edges}
        return set(sol.get_path()).difference(edge_set)

    def all_edge_linked_to_root(self,sol:Solution) -> bool:
        """
            Returns True when the solution has no selected edge that are disconnected from the root node
        """
        return len(self.edges_disconnected_from_root(sol))==0

    def edges_disconnected_from_root(self,sol:Solution)-> set[Edge]:
        """
            Returns all edges disconnected from root 
        """
        disconnected = set(sol.get_path())
        visited = set()
        queue = deque()
        queue.append(self.get_root())

        while len(queue) > 0 and len(disconnected) > 0:
            current = queue.pop()
            if current in visited:
                continue

            visited.add(current)
            for edge in self.get_neighbors(current):
                if (edge.idx().issubset(visited)) or (edge not in disconnected):
                    continue

                disconnected.remove(edge)

                for node in edge.idx().difference((current,)):
                    queue.append(node)
            
        return disconnected

    def all_node_within_maximal_distance(self,sol:Solution) -> bool:
        """
            Returns True when the solution has no node further than the maximal distance H
        """
        return len(self.nodes_out_of_maximal_distance(sol))==0
   
    def nodes_out_of_maximal_distance(self,sol:Solution)-> set[tuple[int, Edge]]:
        """
            Returns all nodes whose distance from root is greater than H 
        """
        distances = set()
        visited = set()
        path = set(sol.get_path())
        queue = [] 

        heapq.heapify(queue)
        heapq.heappush(queue, (0, self.get_root()))

        while len(queue) > 0:
            distance, current = heapq.heappop(queue)
            if current in visited:
                continue
            
            visited.add(current)
            distances.add((distance, current))

            for edge in self.get_neighbors(current):
                if (edge.idx().difference((current,)) in visited) or (edge not in path):
                    continue

                path.remove(edge)

                for node in edge.idx().difference((current,)):
                    heapq.heappush(queue, (distance+1, node))
        
        return set(filter(lambda d: d[0] > self.H, distances))
    
    def is_budget_not_exceeded(self,sol:Solution) -> bool:
        """
            Returns True when the budget B is not exceeded  
        """
        return self.exceeded_budget(sol) <= 0

    def exceeded_budget(self,sol:Solution) -> int:
        """
            Returns the budget exceeded by the solution cost
        """
        return max(0, self.solution_cost(sol) - self.B)
    
    def solution_cost(self, sol: Solution) -> int:
        """
            Compute and return the cost in budget of a solution
        """
        cost = 0
        for edge in filter(lambda x: x is not None, sol.get_path()):
            cost += edge.cost()
        return cost
    
    def solution_value(self, sol: Solution) -> float:
        """
            Compute and return the revenue value of a solution
        """
        revenue = 0
        visited = set()
        for edge in filter(lambda x: x is not None, sol.get_path()):
            n1,n2 = edge.idx()
            if n1 not in visited:
                revenue += n1.revenue()
                visited.add(n1)
            if n2 not in visited:
                revenue += n2.revenue()
                visited.add(n2)

        return revenue

    def solution_value_and_validity(self, sol: Solution) -> tuple[float,bool]:
        """
            Return the revenue and validity of a solution
        """
        return self.solution_value(sol), self.is_valid_solution(sol)

    def generate_distinct_colors(self, num_colors):
        """
            Generates an array of #num_colors colors such that
            the colors are the most distinct possible
        """
        return distinctipy.get_colors(num_colors)

    def visualize_instance(self):
        """
            Show the instance graph
        """
        G = nx.Graph()
        G.add_nodes_from([n for n in self.nodes.values()])
        G.add_edges_from([e.idx() for e in self.edges])
        pos = nx.bfs_layout(G, self.get_root(), align="horizontal")
        pos[self.get_root()] = (0,0)
        if len(self.edges) >= 1000:
            k = 15/np.sqrt(len(G.nodes()))
        else:
            k = 6/np.sqrt(len(G.nodes()))
        pos = nx.spring_layout(G, k=k, pos=pos, fixed=[self.get_root()], seed=38206, scale=10)
        # Nodes colored by cluster
        fig, _ = plt.subplots(figsize=figure_size)

        # nx.draw(G, pos=pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, [self.get_root()], node_shape="s", node_color="red", node_size=750)
        nx.draw_networkx_nodes(G, pos, set(self.profit_nodes.values()).difference((self.get_root(),)), node_shape="v", node_color="orange", node_size=500)
        nx.draw_networkx_nodes(G, pos, {n for n in self.nodes.values()}.difference(self.profit_nodes.values()),)
        
        nx.draw_networkx_edges(G, pos)

        fig.suptitle("Visualisation de l'instance", fontsize=16)
        fig.tight_layout()
        plt.show()
        plt.close()

    def break_path(self, path: list[Edge]) -> list[list[tuple[Node, Node]]]:
        path = set(path)
        multi_path: list[list[tuple[Node, Node]]] = []

        root = self.get_root()
        visited_edge = set()
        split_dir: list[tuple[Edge, Node]] = []
        for edge in filter(lambda x: x in path, self.get_neighbors(root)):
            split_dir.append((edge, root))
            visited_edge.add(edge)
        
        i = 0
        while len(split_dir) > 0 and len(path) > 0:
            splited_edge, root_split = split_dir.pop(0)

            multi_path.append([tuple(splited_edge.idx())])
            path.remove(splited_edge)
            visited_edge.add(splited_edge)

            visited = set()
            queue = []
            queue.extend(splited_edge.idx().difference((root_split,)))

            while len(queue) > 0 and len(path) > 0:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)

                for edge in filter(lambda x: x in path, self.get_neighbors(current)):
                    if edge in visited_edge \
                        or edge.idx().difference((current,)).issubset(visited):
                        continue

                    visited_edge.add(edge)
                    multi_path[i].append(tuple(edge.idx()))
                    path.remove(edge)
                    queue.extend(edge.idx().difference((current,)))

                    
            i += 1

        return multi_path
    
    def visualize_solution(self, sol: Solution):
        """
            Show and save the solution's visualization
        """
        G = nx.Graph()
        G.add_nodes_from([n for n in self.nodes.values()])
        G.add_edges_from([e.idx() for e in self.edges])
        pos = nx.bfs_layout(G, self.get_root(), align="horizontal")
        pos[self.get_root()] = (0,0)
        if len(self.edges) >= 1000:
            k = 15/np.sqrt(len(G.nodes()))
        else:
            k = 6/np.sqrt(len(G.nodes()))
        pos = nx.spring_layout(G, k=k, pos=pos, fixed=[self.get_root()], seed=38206, scale=10)
        # Nodes colored by cluster
        fig, _ = plt.subplots(figsize=figure_size)

        # nx.draw(G, pos=pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, [self.get_root()], node_shape="s", node_color="red", node_size=750)
        nx.draw_networkx_nodes(G, pos, set(self.profit_nodes.values()).difference((self.get_root(),)), node_shape="v", node_color="orange", node_size=500)
        nx.draw_networkx_nodes(G, pos, {n for n in self.nodes.values()}.difference(self.profit_nodes.values()),)
        
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


        multi_path = self.break_path(sol.get_path())
        colors = self.generate_distinct_colors(len(multi_path))
        for i_path, color in zip(multi_path, colors): 
            nx.draw_networkx_edges(G, pos, i_path, edge_color=color, width=15, alpha=0.5)

        revenu = self.solution_value(sol)
        consummed = self.solution_cost(sol)
        fig.suptitle(f"Solution de {self.filepath.stem}\nBudget consommé = {consummed}, Revenu = {revenu}", fontsize=18)
        fig.tight_layout()
        plt.savefig("visualization_"+self.filepath.stem+".png")
        plt.show()
        plt.close()
    
    def save_solution(self, sol: Solution) -> None:
        """
            Saves the solution to a file
        """
        solution_dir = Path(os.path.join(os.path.dirname(__file__),"solutions"))
        if not solution_dir.exists():
            solution_dir.mkdir()

        with open(os.path.join(solution_dir, self.filepath.stem + ".txt"),'w+') as f:
            f.write(f"{sol.T}\n")
            for edge in sol.get_path():
                f.write(f'{" ".join(map(lambda x: str(x.idx()), edge.idx()))}\n')

    
    def read_solution(self, in_file: str) -> Solution:
        """
            Read a solution file
        """
        solution_file = Path(make_universal(in_file))

        with open(solution_file) as f:
            lines = list([[int(x.strip()) for x in x.split(' ') if x.strip() != ''] for x in f.readlines()])
            T = int(lines[0][0])
            path = []

            for line in lines[1:T+1]:
                n1,n2 = tuple(map(int,line))
                for edge in self.edges:
                    if edge.has_node(n1) and edge.has_node(n2):
                        path.append(edge)

        assert(T == len(path))
        return Solution(path)
import os

import matplotlib.pyplot as plt
import numpy as np

from typing import Iterable
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

class Solution:
    def __init__(self, sequence: Iterable[int]):
        self.__sequence = list(sequence)

    def get_sequence(self) -> list[int]:
        return self.__sequence

    def __repr__(self):
        return 'Solution'+str(self.__sequence)
    
class Instance:
    def __init__(self,in_file: str):
        self.filepath = Path(make_universal(in_file))
        assert(self.filepath.exists())
        with open(self.filepath) as f:
            lines = list([[x.strip() for x in x.split(' ') if x.strip() != ''] for x in f.readlines()])
            
            self.J = int(lines[0][0])
            self.C = int(lines[1][0])

            self.demands = [[int(k) for k in lines[2 + i]] for i in range(self.C)]

            self.H = int(lines[2 + self.C][0])

            self.transition_costs = [[int(k) for k in lines[3 + self.C + i]] for i in range(self.C)]

    def order(self, j: int, c: int) -> bool:
        """
            Return true if there is an item c ordered for time j

        :param j: time stamp (int) of the wanted time
        :param c: id (int) of the item
        """
        assert(j >= 0 and j < self.J)
        assert(c >= -1 and c < self.C)

        if c == -1:
            return False
        
        return self.demands[c][j]

    def transition_cost(self, c1, c2):
        """
            Return transition cost from item c1 to item c2

        :param c1: id (int) of the 1st item
        :param c2: id (int) of the 2nd item
        """
        if c1 < 0 or c2 < 0:
            return 0

        return self.transition_costs[c1][c2]

    def is_valid_solution(self,sol:Solution) -> bool:
        """ 
            Returns True when the solution is valid
        """
        return self.all_asked_products(sol) and self.no_late_deliveries(sol)


    def all_asked_products(self,sol:Solution) -> bool:
        """
            Returns True when the solution produce all demanded products
        """
        return len(self.production_difference(sol))==0
   
    def production_difference(self,sol:Solution)-> set[tuple[int, int]]:
        """
            Returns all products whose exact demands through all the planning is not exactly met.
            If the second item of the tuple is positive, the solution produce too much of that item.
            If the second item of the tuple is negative, the solution don't produce enough of that item.
        """
        S = sol.get_sequence()
        product_diff = []
        for c in range(self.C):
            s = [j for j in range(self.J) if self.order(j, c)]
            p = [j for j in range(self.J) if S[j] == c]
            product_diff.append((c, len(p) - len(s)))
        
        return set(filter(lambda p: p[1] != 0, product_diff))
    
    def no_late_deliveries(self,sol:Solution) -> bool:
        """
            Returns True when the budget B is not exceeded  
        """
        return len(self.late_deliveries(sol)) == 0

    def late_deliveries(self,sol:Solution) -> set[tuple[int, int]]:
        """
            Returns all days where the demands are not met by the solution.
            First item of tuple is the day, second item is the late product
        """
        S = sol.get_sequence()
        stock = [0 for _ in range(self.C)]
        late_demand = set()
        for j in range(self.J):
            c_prod = S[j]

            if c_prod > -1:
                stock[c_prod] += 1

            for c in range(self.C):
                if self.order(j,c):
                    if stock[c] > 0:
                        stock[c] -= 1
                    else:
                        late_demand.add((j,c))
            
        return late_demand
    
    def solution_cost(self, sol: Solution) -> int:
        """
            Compute and return the cost of a solution
        """
        return self.solution_storage_cost(sol) + self.solution_transition_cost(sol)
    

    def solution_storage_cost(self, sol: Solution) -> int:
        """
            Compute the cost of storage in a solution
        """
        S = sol.get_sequence()
        sum_date = 0
        for c in range(self.C):
            cnt = 0
            for j in range(self.J):
                if S[j] == c:
                    cnt += 1
                if self.order(j,c):
                    cnt -= 1
                sum_date += cnt

        return self.H * sum_date

    def solution_transition_cost(self, sol: Solution) -> int :
        """
            Compute the cost of transitions in a solution
        """
        S = sol.get_sequence()
        sum_tran = 0
        last_c = -1
        for j in range(self.J):
            c = S[j]
            if c > -1:
                if last_c > -1:
                    sum_tran += self.transition_cost(last_c, c)
                last_c = c

        return sum_tran


    def solution_cost_and_validity(self, sol: Solution) -> tuple[float,bool]:
        """
            Return the cost and validity of a solution
        """
        return self.solution_cost(sol), self.is_valid_solution(sol)


    def visualize_instance(self):
        """
            Show the instance
        """
        fig, ax = plt.subplots(2, figsize=figure_size)

        x = [i +1 for i in range(self.J)]
        prev = [0] * self.J
        for i in range(self.C):
            ax[0].bar(x, self.demands[i], 0.5, bottom=prev, label='Prod ' + str(i+1))
            for j in range(self.J):
                prev[j] += self.demands[i][j]
        ax[0].set_yticks([])
        ax[0].set_xticks(np.arange(0, self.J+1, 5))
        ax[0].legend()
        ax[0].set_title("Demandes de livraisons")


        ax[1].set_title("Coûts de transition")
        ax[1].matshow(self.transition_costs, cmap='seismic')
        for (i, j), z in np.ndenumerate(self.transition_costs):
            ax[1].text(j, i, '{}'.format(z), ha='center', va='center', color="black",
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        fig.suptitle("Visualisation de l'instance", fontsize=16)
        fig.tight_layout()
        plt.show()

    
    def visualize_solution(self, sol: Solution):
        """
            Show and save the solution's visualization
        """
        fig, ax = plt.subplots(3, figsize=figure_size)
        S = sol.get_sequence()
        x = [i +1 for i in range(self.J)]
        prev = [0] * self.J
        for i in range(self.C):
            ax[0].bar(x, self.demands[i], 0.5, bottom=prev, label='Prod ' + str(i+1))
            for j in range(self.J):
                prev[j] += self.demands[i][j]
        ax[0].set_yticks([])
        ax[0].set_xticks(np.arange(0, self.J+1, 5))
        ax[0].legend()
        ax[0].set_title("Demandes de livraisons")

        for i in range(self.C):
            prod_y = [1 if j == i else 0 for j in S]
            ax[1].bar(x, prod_y, 0.5, label='Prod ' + str(i))
        ax[1].set_title("Planification (coût total des transitions = {0})".format(self.solution_transition_cost(sol)))
        ax[1].set_yticks([])
        ax[1].set_xticks(np.arange(0, self.J+1, 5))

        x = [i for i in range(self.J+1)]
        ax[2].set_title("Produits en stock (coût total du stockage = {0})".format(self.solution_storage_cost(sol)))
        for c in range(self.C):
            storage_y = [0 for _ in range(self.J)]
            for i in range(self.J):
                if i != 0:
                    storage_y[i] += storage_y[i - 1]
                if S[i] == c:
                    storage_y[i] += 1
                if self.order(i,c) and storage_y[i] > 0:
                    storage_y[i] -= 1
            ax[2].plot(x, [0] + storage_y)
        ax[2].set_xticks(np.arange(0, self.J+1, 5))


        fig.suptitle("Solution de l'instance", fontsize=16)
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
            f.write(f'{" ".join(map(str,sol.get_sequence()))}')

    
    def read_solution(self, in_file: str) -> Solution:
        """
            Read a solution file
        """
        solution_file = Path(make_universal(in_file))

        with open(solution_file) as f:
            sequence = [int(i.strip()) for i in f.read().split(" ") if i.strip() != '']

        return Solution(sequence)
    
if __name__ == "__main__":
    inst = Instance("./instances/instance_B_100_10_10.txt")

    sol: Solution = inst.read_solution("./solutions/instance_B_100_10_10.txt")
    print(inst.production_difference(sol))
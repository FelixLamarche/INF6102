from utils import Instance, Solution

import copy
import numpy as np

class CustomSolution(Solution):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, revenue):
        super().__init__(idx, revenue)

class SolverAdvanced :
    def __init__(self, instance : Instance):
        self.instance = instance
        self.J = self.instance.J
        self.C = self.instance.C
        self.H = self.instance.H
        self.demands = self.instance.demands
        self.transition_costs = self.instance.transition_costs

        self.seq = [-1 for j in range(self.J)]
        self.cost = 0
        self.stocks = [[0 for k in range(self.J)] for i in range(self.C)]

        self.best_seq = []
        self.best_cost = 1e10

    def local_search(self) :
        self.seq = self.solve_heuristic()
        self.cost = self.compute_cost()
        self.compute_stocks()
        self.best_seq = self.seq
        self.best_cost = self.cost

        improved = True
        already_moved = set()

        while improved :
            improved = False

            best_swap = None
            best_swap_cost = 1e10

            for j1 in range(self.J) :
                for j2 in range(j1+1, self.J) :
                    product1, product2 = self.seq[j1], self.seq[j2]

                    # If it's the same product or nothing is produced, continue
                    if j1 == j2 or product1 == product2 or product1 == -1 or product2 == -1 :
                        continue

                    new_cost = self.test_swap(j1, j2)
                    # if not self.test_swap(j1, j2) :
                    if not new_cost :
                        # print(f"PB timing consommation for swap {j1} produit {product1+1} and swap {j2} produit {product2+1}")
                        continue
                    
                    # swap_cost = self.calculate_cost_swap(product1, product2, j1, j2)
                    if new_cost < best_swap_cost and j1 not in already_moved :
                        best_swap = (j1, j2)
                        best_swap_cost = new_cost

            if best_swap :
                j1, j2 = best_swap
                self.swap(j1, j2)
                self.cost = best_swap_cost
                self.compute_stocks()
                print(f"Jour {j1} et jour {j2} swappés")
                already_moved.add(j1)
                improved = True

            self.best_seq = self.seq
            self.best_cost = self.cost

    def solve_heuristic(self) :
        """
        Heuristics that completes the planning backwards in a greedy way
        Gives 3.57/8 and is very fast
        """
        needed = []
        self.seq = [-1 for i in range(self.J)]
        for j in range(self.J-1, -1, -1) :
            for c in range(self.C) :
                if self.instance.order(j,c) :
                    needed.append(c)
            if needed :
                # produced = np.random.choice(needed)
                produced = self.choose_best_product(needed, j)
                needed.remove(produced)
                self.seq[j] = produced
        return self.seq
    
    def choose_best_product(self, needed, day) :
        needed_set = set(needed) #do not test twice the same product
        best_cost = 1e10
        best_product = None
        for product in needed_set :
            cost = 0
            #Compute storage cost
            for j in range(day, self.J) :
                if self.demands[product][j] > 0 :
                    break
                cost += self.H
            #Compute transition cost
            c_after, j_after = -1, day
            while c_after == -1 and j_after < self.J-1 :
                c_after, j_after = self.seq[j_after+1], j_after+1
            if j_after == self.J-1 :
                best_product = np.random.choice(needed)
                continue
            cost += self.transition_costs[product][c_after]

            if cost < best_cost :
                best_cost = cost
                best_product = product
            
        return best_product
            
    def test_swap(self, j1, j2) :
        # Check if swap is possible in terms of demand and stocks
        product1 = self.seq[j1]
        product2 = self.seq[j2]

        for j in range(j1, j2):
            if self.instance.order(j,product1) :
                if self.stocks[product1][j] -1 <= 0 :
                    return False
                
        # Check if the swap improves the solution
        current_seq = copy.deepcopy(self.seq)
        current_cost = self.cost
        self.seq[j1] = product2
        self.seq[j2] = product1
        new_cost = self.compute_cost()
        if new_cost >= current_cost :
            # Go back to initial seq
            self.seq = current_seq
            self.cost = current_cost
            return False
        
        # Go back to initial seq
        self.seq = current_seq
        self.cost = current_cost
        
        return new_cost

    def calculate_cost_swap(self, product1, product2, j1, j2) :
        """
        Calculates the cost of swapping product made in day j1 and product made in day j2
        """
        # Cas non géré : si c'est -1
        product_before_p1 = self.seq[j1-1]
        product_before_p2 = self.seq[j2-1]
        product_after_p1 = self.seq[j1+1]
        product_after_p2 = self.seq[j2+1]

        # Transition costs
        old_transition_costs = self.transition_costs[product_before_p1][product1] + self.transition_costs[product1][product_after_p1] + self.transition_costs[product_before_p2][product2] + self.transition_costs[product2][product_after_p2]
        new_transition_costs = self.transition_costs[product_before_p1][product2] + self.transition_costs[product2][product_after_p1] + self.transition_costs[product_before_p2][product1] + self.transition_costs[product1][product_after_p2]
        
        return new_transition_costs - old_transition_costs

    def swap(self, j1, j2) :
        """
        Swaps product made in day j1 and product made in day j2
        DOES NOT UPDATE COST
        """
        product1 = self.seq[j1]
        product2 = self.seq[j2]
        self.seq[j1] = product2
        self.seq[j2] = product1


    def naive_solution(self) :
        """ Returns naive solution
        Returns:
            seq : a solution
        """

        seq = [-1 for _ in range(self.J)]
        pos = 0


        for j in range(self.J):
            for c in range(self.C):
                if self.instance.order(j,c):
                    seq[pos] = c
                    pos += 1

        self.seq = seq
        self.cost = self.compute_cost()
        self.compute_stocks()

    def compute_cost(self) :
        # Compute storage cost
        sum_date = 0
        for c in range(self.C):
            cnt = 0
            for j in range(self.J):
                if self.seq[j] == c:
                    cnt += 1
                if self.instance.order(j,c):
                    cnt -= 1
                sum_date += cnt
        # Compute transition costs
        sum_tran = 0
        last_c = -1
        for j in range(self.J):
            c = self.seq[j]
            if c > -1:
                if last_c > -1:
                    # sum_tran += self.transition_costs(last_c, c)
                    sum_tran += self.transition_costs[last_c][c]
                last_c = c

        return self.H * sum_date + sum_tran

    def compute_stocks(self) :
        self.stocks = [[0 for k in range(self.J)] for i in range(self.C)]
        for c in range(self.C):
            # storage_y = [0 for _ in range(self.J)]
            for j in range(self.J):
                if j != 0:
                    self.stocks[c][j] += self.stocks[c][j-1]
                    # storage_y[i] += storage_y[i - 1]
                if self.seq[j] == c:
                    # storage_y[i] += 1
                    self.stocks[c][j] += 1
                if self.instance.order(j,c) and self.stocks[c][j] > 0:
                    # storage_y[i] -= 1
                    self.stocks[c][j] -= 1

        
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """
    solver = SolverAdvanced(instance)
    # solver.best_seq = solver.solve_heuristic()
    solver.local_search()
    
    sol = Solution(solver.best_seq)

    return sol

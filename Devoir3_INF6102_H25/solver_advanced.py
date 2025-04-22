# By:
# Felix Lamarche - 2077446
# Auriane Peter–Hemon - 2310513
from utils import Instance, Solution

import copy
import itertools as it
import numpy as np
import random
import time

class CustomSolution(Solution):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, revenue):
        super().__init__(idx, revenue)

class SolverAdvanced :
    def __init__(self, instance : Instance, timeout_time : float) :
        self.instance = instance
        self.J = self.instance.J
        self.C = self.instance.C
        self.H = self.instance.H
        self.demands = self.instance.demands
        self.day_demands = []
        self.compute_day_demands()
        self.transition_costs = self.instance.transition_costs

        self.seq = [-1 for j in range(self.J)]
        self.cost = 0
        self.stocks = [[0 for k in range(self.J)] for i in range(self.C)]

        self.best_seq = []
        self.best_cost = 1e10

        self.timeout_time = timeout_time

    def first_or_k_opt_move(self, k):
        """
        OR-k-opt (relocate k elements)
        """
        best_move = None
        best_move_cost = 1e10

        days = [i for i in range(self.J)]
        random.shuffle(days)

        for i in days:
            if i+k > self.J:
                continue

            subseq = self.seq[i:i+k]
            if k <= 3:
                subseq_orders = [list(subseq_order) for subseq_order in it.permutations(subseq)]
            else:
                subseq_orders = [subseq]

            for j in days:
                if j >= i and j < i + k or j + k > self.J:
                    continue

                # Temporarily relocate product i -> j
                insert_pos = j
                if j > i:
                    insert_pos -= k
                new_seq = self.seq[:i] + self.seq[i+k:] # Remove subsequence

                for subseq_order in subseq_orders:
                    new_seq = new_seq[:insert_pos] + subseq_order + new_seq[insert_pos:] # Insert

                    if not self.test_seq_stocks(new_seq) :
                        continue

                    new_cost = self.test_compute_cost(new_seq)
                    if new_cost >= self.cost :
                        continue

                    if new_cost < best_move_cost:
                        best_move = (i, j, subseq_order)
                        best_move_cost = new_cost
                if best_move:
                    break
            if best_move:
                break

        if best_move:
            i, j, subseq = best_move
            insert_pos = j
            if j > i:
                insert_pos -= k
            self.seq = self.seq[:i] + self.seq[i+k:]
            self.seq = self.seq[:insert_pos] + subseq + self.seq[insert_pos:]
            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Best Moved products {subseq} from day {i} to day {j}, k={k}")

    def best_or_k_opt_move(self, k):
        """
        OR-k-opt (relocate k elements)
        """
        best_move = None
        best_move_cost = 1e10

        for i in range(self.J):
            if i+k > self.J :
                continue

            subseq = self.seq[i:i+k]

            for j in range(self.J - k + 1):
                if j >= i  and j < i + k :
                    continue

                # Temporarily relocate product i -> j
                insert_pos = j
                if j > i:
                    insert_pos -= k
                new_seq = self.seq[:i] + self.seq[i+k:]
                new_seq = new_seq[:insert_pos] + subseq + new_seq[insert_pos:]

                if not self.test_seq_stocks(new_seq) :
                    continue

                new_cost = self.test_compute_cost(new_seq)
                if new_cost >= self.cost :
                    continue

                if new_cost < best_move_cost:
                    best_move = (i, j)
                    best_move_cost = new_cost

        if best_move:
            i, j = best_move
            insert_pos = j
            if j > i:
                insert_pos -= k
            subseq = self.seq[i:i+k]
            self.seq = self.seq[:i] + self.seq[i+k:]
            self.seq = self.seq[:insert_pos] + subseq + self.seq[insert_pos:]
            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Best Moved products {subseq} from day {i} to day {j}, k={k}")

    def first_or_opt_move_1_pop_insert(self):
        """
        OR-1-opt (relocate one element)
        Pops the product and inserts it at its new position pushing the products to the right
        """
        best_move = None
        best_move_cost = 1e10

        days = [i for i in range(self.J)]
        random.shuffle(days)

        for i in days:
            for j in days:
                if i == j:
                    continue

                new_seq = self.seq[:]
                product = new_seq.pop(i)
                new_seq.insert(j, product)
                
                if not self.test_seq_stocks(new_seq) :
                    continue

                new_cost = self.test_compute_cost(new_seq)

                if new_cost < best_move_cost and new_cost < self.cost:
                    best_move = (i, j)
                    best_move_cost = new_cost
                    break
            if best_move:
                break

        if best_move:
            i, j = best_move
            product = self.seq.pop(i)
            self.seq.insert(j, product)
            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Moved product {product} from day {i} to day {j}, pop-insert")

    def best_or_opt_move_1_pop_insert(self):
        """
        OR-1-opt (relocate one element)
        Pops the product and inserts it at its new position pushing the products to the right
        """
        best_move = None
        best_move_cost = 1e10

        for i in range(self.J):
            # We will move the product i through all days by swapping it with the product of the next day
            new_seq = self.seq[:]
            # We set it first, then swap it with the next day's product for every day
            product = new_seq.pop(i)
            new_seq.insert(0, product)
            for j in range(self.J):
                # Swap product with the next day's product
                if j > 0:
                    temp = new_seq[j - 1]
                    new_seq[j - 1] = new_seq[j]
                    new_seq[j] = temp
                
                if not self.test_seq_stocks(new_seq) :
                    continue

                new_cost = self.test_compute_cost(new_seq)

                if new_cost < best_move_cost and new_cost < self.cost:
                    best_move = (i, j)
                    best_move_cost = new_cost

        if best_move:
            i, j = best_move
            product = self.seq.pop(i)
            self.seq.insert(j, product)
            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Best Moved product {product} from day {i} to day {j}, pop-insert")

    def first_or_opt_move_1_insert_pop(self):
        """
        OR-1-opt (relocate one element)
        Places a -1 where the product was and inserts it at its new position trying to push the products to the left 
        by trying to see if an empty space is available on their left
        """
        best_move = None
        best_move_cost = 1e10

        days = [i for i in range(self.J)]
        random.shuffle(days)

        for i in days:
            for j in days:
                if i == j:
                    continue

                new_seq = self.seq[:]
                product = new_seq[i]
                new_seq[i] = -1
                new_seq.insert(j, product)

                did_pop = False
                for k in range(j, -1, -1):
                    if new_seq[k] == -1:
                        new_seq.pop(k)
                        did_pop = True
                        break

                if not did_pop:
                    continue
                
                if not self.test_seq_stocks(new_seq) :
                    continue

                new_cost = self.test_compute_cost(new_seq)

                if new_cost < best_move_cost and new_cost < self.cost:
                    best_move = (i, j)
                    best_move_cost = new_cost
                    break
            if best_move:
                break

        if best_move:
            i, j = best_move
            product = self.seq[i]
            self.seq[i] = -1
            self.seq.insert(j, product)
            for k in range(j, -1, -1):
                if self.seq[k] == -1:
                    self.seq.pop(k)
                    break

            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Moved product {product} from day {i} to day {j}, insert-pop")

    def best_or_opt_move_1_insert_pop(self):
        """
        OR-1-opt (relocate one element)
        Places a -1 where the product was and inserts it at its new position trying to push the products to the left 
        by trying to see if an empty space is available on their left
        """
        best_move = None
        best_move_cost = 1e10

        for i in range(self.J):
            for j in range(self.J):
                if i == j:
                    continue

                new_seq = self.seq[:]
                product = new_seq[i]
                new_seq[i] = -1
                new_seq.insert(j, product)

                did_pop = False
                for k in range(j, -1, -1):
                    if new_seq[k] == -1:
                        new_seq.pop(k)
                        did_pop = True
                        break

                if not did_pop:
                    continue
                
                if not self.test_seq_stocks(new_seq) :
                    continue

                new_cost = self.test_compute_cost(new_seq)

                if new_cost < best_move_cost and new_cost < self.cost:
                    best_move = (i, j)
                    best_move_cost = new_cost

        if best_move:
            i, j = best_move
            product = self.seq[i]
            self.seq[i] = -1
            self.seq.insert(j, product)
            for k in range(j, -1, -1):
                if self.seq[k] == -1:
                    self.seq.pop(k)
                    break

            self.cost = best_move_cost
            self.compute_stocks()
            # print(f"Best Moved product {product} from day {i} to day {j}, insert-pop")

    def first_2_opt_swap(self):
        """
        Best 2-opt swap local search
        """
        days = [i for i in range(self.J)]
        random.shuffle(days)

        best_swap = None
        best_swap_cost = 1e10
        for j1 in days:
            for j2 in days[j1+1:]:
                product1, product2 = self.seq[j1], self.seq[j2]

                if product1 == product2:
                    continue
                
                new_cost = self.test_swap(min(j1, j2), max(j1, j2))
                if not new_cost:
                    continue
                
                if new_cost < self.cost:
                    best_swap = (j1, j2)
                    best_swap_cost = new_cost
                    break
            if best_swap:
                break

        if best_swap:
            j1, j2 = best_swap
            product1 = self.seq[j1]
            product2 = self.seq[j2]
            self.seq[j1] = product2
            self.seq[j2] = product1
            self.cost = best_swap_cost # Updates cost
            self.compute_stocks()   # Updates stocks

    def best_2_opt_swap(self):
        """
        Best 2-opt swap local search
        """
        best_swap = None
        best_swap_cost = 1e10

        for j1 in range(self.J) :
            for j2 in range(j1+1, self.J) :
                product1, product2 = self.seq[j1], self.seq[j2]

                if product1 == product2:
                    continue

                new_cost = self.test_swap(j1, j2)
                if not new_cost:
                    continue
                
                if new_cost < best_swap_cost:
                    best_swap = (j1, j2)
                    best_swap_cost = new_cost

        if best_swap :
            j1, j2 = best_swap
            product1 = self.seq[j1]
            product2 = self.seq[j2]
            self.seq[j1] = product2
            self.seq[j2] = product1
            self.cost = best_swap_cost # Updates cost
            self.compute_stocks()   # Updates stocks


    def solve_VND(self):
        """
        Does a variable neighborhood descent
        Reduced variable ngeighborhood search (takes random instead of best)
        Results: 6.89/8 and is not fast
        """
        # Keep doing local search until no improvement is found
        neighborhoods_without_improvement = 0
        NB_NEIGHBORHOODS = 12
        while neighborhoods_without_improvement < NB_NEIGHBORHOODS and time.time() < self.timeout_time:
            prev_cost = self.cost

            # Search a neighborhood
            if neighborhoods_without_improvement == 0:
                self.first_or_opt_move_1_pop_insert()
            elif neighborhoods_without_improvement == 1:
                self.first_or_opt_move_1_insert_pop()
            elif neighborhoods_without_improvement == 2:
                self.first_2_opt_swap()
            else:
                k = neighborhoods_without_improvement - 1
                self.first_or_k_opt_move(k)

            # Check if we improved the solution
            # If we have not improved the solution, we go to the next neighborhood
            # If we have improved the solution, we reset the counter
            if self.cost < prev_cost:
                neighborhoods_without_improvement = 0
            else:
                neighborhoods_without_improvement += 1

            if self.cost < self.best_cost:
                self.best_cost = self.cost
                self.best_seq = self.seq[:]
                # print(f"New best cost: {self.best_cost}\n")

    def skewed_VNS(self, alpha) :
            """
            Skewed version of the VNS. Is quite slow so does not perform well
            Hill-climbing instead of VND in the loop
            """
            neighborhoods_without_improvement = 0
            NB_NEIGHBORHOODS = 8
            while neighborhoods_without_improvement < NB_NEIGHBORHOODS and time.time() < self.timeout_time:
                previous_seq = self.seq[:]
                prev_cost = self.cost
                # Random selection of a neighbor
                if neighborhoods_without_improvement == 0 :
                    self.first_or_opt_move_1_pop_insert()
                elif neighborhoods_without_improvement == 1 :
                    self.first_or_opt_move_1_insert_pop()
                elif neighborhoods_without_improvement == 2 :
                    self.first_2_opt_swap()
                else : 
                    k = neighborhoods_without_improvement - 1
                    self.first_or_k_opt_move(k)

                # Hill climbing
                #if neighborhoods_without_improvement == 0 :
                #    self.solve_move_k(1)
                #else :
                #    self.solve_move_k(neighborhoods_without_improvement)

                # Neighborhood change
                if self.cost < prev_cost + alpha*(prev_cost - self.cost) :
                    neighborhoods_without_improvement = 0
                    if self.cost > prev_cost :
                        print(f"Mauvaise solution acceptée")
                else :
                    self.seq = previous_seq[:]
                    neighborhoods_without_improvement += 1

    def solve_move_1(self):
        """
        Solve with relocate 1-element local search
        Results: 5.22/8
        """
        # Keep doing local search until no improvement is found
        prev_cost = -1
        while prev_cost != self.cost and time.time() < self.timeout_time:
            prev_cost = self.cost
            self.or_opt_move_1()

            if self.cost < self.best_cost:
                self.best_cost = self.cost
                self.best_seq = self.seq[:]
                # print(f"New best cost: {self.best_cost}")

    def solve_move_k(self, k):
        """
        Solve with relocate k-element local search
        Results K=1: 5.21/8
        Results K=2: 4.77/8
        Results K=3: 4.32/8
        Results K=4  4.39/8
        Results K=5  4.00/8
        Results K=6  3.83/8
        """
        # Keep doing local search until no improvement is found
        prev_cost = -1
        while prev_cost != self.cost and time.time() < self.timeout_time:
            prev_cost = self.cost
            self.first_or_k_opt_move(k)

            if self.cost < self.best_cost:
                self.best_cost = self.cost
                self.best_seq = self.seq[:]
                # print(f"New best cost: {self.best_cost}")
            

    def set_initial_seq(self, seq):
        self.seq = seq
        self.cost = self.compute_cost()
        self.compute_stocks()
        if self.cost < self.best_cost:
            self.best_seq = self.seq[:]
            self.best_cost = self.cost

    def solve_heuristic(self) :
        """
        Heuristics that completes the planning backwards in a greedy way
        Useful as an initial solution
        Gives 3.57/8 and is very fast
        """
        needed = []
        self.seq = [-1 for i in range(self.J)]
        for j in range(self.J-1, -1, -1) :
            for c in range(self.C) :
                if c in self.day_demands[j]:
                    needed.append(c)
            if needed :
                # produced = np.random.choice(needed)
                produced = self.choose_best_product(needed, j)
                needed.remove(produced)
                self.seq[j] = produced
        return self.seq
    
    def solve_naive_random(self):
        """
        Random solution
        Useful as an initial solution
        Results: 0.0/8
        """
        seq = [-1 for _ in range(self.J)]
        pos = 0

        for j in range(self.J):
            demands = self.day_demands[j][:]
            random.shuffle(demands)
            for c in demands:
                seq[pos] = c
                pos += 1

        self.seq = seq
        self.cost = self.compute_cost()
        self.compute_stocks()
        return seq

    def naive_solution(self) :
        """ Returns naive solution
        Useful as an initial solution
        Results: 0.0/8
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
        return seq

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
        """
        Checks if a swap between j1 and j2 is valid in terms of demand and if it improves the current solution
        Requires that: j1 < j2
        """
        # Check if swap is possible in terms of demand and stocks
        product1 = self.seq[j1]
        product2 = self.seq[j2]

        for j in range(j1, j2):
            if product1 in self.day_demands[j]:
                if self.stocks[product1][j] -1 < 0 :
                    return False
                
        # Check if the swap improves the solution
        current_seq = self.seq[:]
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

    def test_seq_stocks(self, seq) :
        """
        Checks if a sequence is valid in termes of demand
        """
        stock = [0 for _ in range(self.C)]
        for j in range(self.J):
            c_prod = seq[j]

            if c_prod > -1:
                stock[c_prod] += 1

            for c in self.day_demands[j]:
                stock[c] -= 1
                if stock[c] < 0:
                    return False
            
        return True

    def compute_cost(self) :
        # Compute storage cost
        sum_date = 0
        for c in range(self.C):
            cnt = 0
            for j in range(self.J):
                if self.seq[j] == c:
                    cnt += 1
                if c in self.day_demands[j]:
                    cnt -= 1
                sum_date += cnt
        # Compute transition costs
        sum_tran = 0
        last_c = -1
        for j in range(self.J):
            c = self.seq[j]
            if c > -1:
                if last_c > -1:
                    sum_tran += self.transition_costs[last_c][c]
                last_c = c

        return self.H * sum_date + sum_tran

    def test_compute_cost(self, test_seq) :
        # Compute storage cost
        sum_date = 0
        cnt = 0
        for j in range(self.J):
            c = test_seq[j]
            if c > -1:
                cnt += 1
            cnt -= len(self.day_demands[j])
            sum_date += cnt
        # Compute transition costs
        sum_tran = 0
        last_c = -1
        for j in range(self.J):
            c = test_seq[j]
            if c > -1:
                if last_c > -1:
                    sum_tran += self.transition_costs[last_c][c]
                last_c = c

        return self.H * sum_date + sum_tran

    def compute_stocks(self) :
        self.stocks = [[0 for k in range(self.J)] for i in range(self.C)]
        for c in range(self.C):
            for j in range(self.J):
                if j != 0:
                    self.stocks[c][j] += self.stocks[c][j-1]
                if self.seq[j] == c:
                    self.stocks[c][j] += 1
                if self.instance.order(j,c) and self.stocks[c][j] > 0:
                    self.stocks[c][j] -= 1

    def compute_day_demands(self):
        """
        Set the day demands for each product
        """
        self.day_demands = []
        for j in range(self.J):
            day_demand = []
            for c in range(self.C):
                if self.instance.order(j,c) :
                    day_demand.append(c)
            self.day_demands.append(day_demand)

        
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with an iterator on Edge object
    """
    #TIME_LIMIT_SEC = 60
    TIME_LIMIT_SEC = get_time_limit(instance)
    TIME_LIMIT_MARGIN_SEC = 15

    timeout_time = time.time() + TIME_LIMIT_SEC - TIME_LIMIT_MARGIN_SEC

    best_seq = None
    best_seq_cost = 1e10

    while time.time() < timeout_time:
        solver = SolverAdvanced(instance, timeout_time)
        # Set initial solution
        init_seq = solver.solve_heuristic()
        #init_seq = solver.solve_naive_random()
        #init_seq = solver.naive_solution()
        solver.set_initial_seq(init_seq)

        # solver.best_seq = solver.solve_heuristic()
        #solver.solve_move_1()
        #solver.solve_move_k(6)
        #solver.solve_hill_climbing_2_opt()
        solver.solve_VND()
        #solver.skewed_VNS(0.90)

        if solver.best_cost < best_seq_cost:
            best_seq_cost = solver.best_cost
            best_seq = solver.best_seq[:]
    
    
    sol = Solution(best_seq)

    return sol

def get_time_limit(instance:Instance) -> float:
    five_minutes_instances = ["instance_A", "instance_B", "instance_C", "instance_D"]
    for name in five_minutes_instances:
        if instance.filepath.name.lower().find(name.lower()) != -1:
            return 300
    # else return 10 minutes
    return 600
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

    def or_k_opt(self, k):
        """
        OR-k-opt (relocate k elements)
        Does not work for k>1
        """
        self.seq = self.solve_heuristic()
        self.cost = self.compute_cost()
        self.compute_stocks()
        print(self.seq, self.cost)

        best_move = None
        best_move_cost = 1e10

        for i in range(self.J):
            if i+k > self.J :
                continue

            subseq = self.seq[i:i+k]

            if all(p == -1 for p in subseq) :
                continue

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
            print(f"Moved products {subseq} from day {i} to day {j}")

        print(self.seq, self.cost)
        self.best_seq = self.seq

    def or_opt_local_search(self):
        """
        OR-1-opt (relocate one element)
        """
        self.seq = self.solve_heuristic()
        self.cost = self.compute_cost()
        self.compute_stocks()

        best_move = None
        best_move_cost = 1e10

        for i in range(self.J):
            product = self.seq[i]

            if product == -1 :
                continue

            for j in range(self.J):
                if i == j :
                    continue

                # Temporarily relocate product i -> j
                new_seq = self.seq[:]
                removed = new_seq.pop(i)
                new_seq.insert(j, removed)

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
            product = self.seq.pop(i)
            self.seq.insert(j, product)
            self.cost = best_move_cost
            self.compute_stocks()
            print(f"Moved product {product} from day {i} to day {j}")

        print(self.seq, self.cost)

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

    def calculate_delta_holding_cost(self, i, j) :
        """
        Compute the change in holding cost when moving the product at index i to index j.
        """
        if i == j:
            return 0

        seq_before = self.seq
        seq_after = seq_before[:]
        product = seq_after.pop(i)
        seq_after.insert(j, product)
        idx_start = min(i, j)
        idx_end = max(i, j)

        # Calculate delta holding cost
        holding_cost_before = 0
        holding_cost_after = 0

        for c in range(self.C):
            stock_before = 0
            stock_after = 0
            for day in range(idx_start, idx_end+1):
                if seq_before[day] == c:
                    stock_before += 1
                if seq_after[day] == c:
                    stock_after += 1
                if self.instance.order(day, c):
                    stock_before -= 1
                    stock_after -= 1
                holding_cost_before += stock_before
                holding_cost_after += stock_after 

        delta_holding = self.H * (holding_cost_after - holding_cost_before)

        return delta_holding
    
    def calculate_delta_transition_costs(self, i, j) :
        """
        DOES NOT WORK
        """
        if i == j:
            return 0

        seq = self.seq[:]
        product = seq[i]

        old_tran = 0
        new_tran = 0

        prev_i = self.get_prev_idx(self.seq, i)
        next_i = self.get_next_idx(self.seq, i)
        prev_j = self.get_prev_idx(self.seq, j)

        if prev_i is not None:
            old_tran += self.transition_costs[self.seq[prev_i]][product]
        if next_i is not None:
            old_tran += self.transition_costs[product][self.seq[next_i]]
        if prev_j is not None :
            old_tran += self.transition_costs[self.seq[prev_j]][self.seq[j]]

        if prev_i and next_i :
            new_tran += self.transition_costs[self.seq[prev_i]][self.seq[next_i]]
        if prev_j is not None :
            new_tran += self.transition_costs[self.seq[prev_j]][product]
        new_tran += self.transition_costs[self.seq[j]][product]

        return new_tran - old_tran
    
    def get_prev_idx(self, seq, idx):
        """
        For a given idx, gives the closest index before where something is produced
        Used to compute transition costs
        """
        for i in range(idx - 1, -1, -1):
            if seq[i] != -1:
                return i
        return None

    def get_next_idx(self, seq, idx) :
        """
        For a given idx, gives the closest index after where something is produced
        Used to compute transition costs
        """
        for i in range(idx + 1, self.J):
            if seq[i] != -1:
                return i
        return None
            
    def test_seq_stocks(self, seq) :
        """
        Checks if a sequence is valid in termes of demand
        """
        stock = [0 for _ in range(self.C)]
        for j in range(self.J):
            c_prod = seq[j]

            if c_prod > -1:
                stock[c_prod] += 1

            for c in range(self.C):
                if self.instance.order(j,c):
                    if stock[c] <= 0:
                        return False
            
        return True

    def hill_climbing_2_opt(self) :
        """
        Does a hill climbing resolution using 2-opt local search
        Gives 4.74/8 and is very fast
        """
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
                    if not new_cost :
                        continue
                    
                    if new_cost < best_swap_cost and j1 not in already_moved :
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
                print(f"Day {j1} and day {j2} swapped")
                already_moved.add(j1)
                improved = True

            self.best_seq = self.seq
            self.best_cost = self.cost

    def test_swap(self, j1, j2) :
        """
        Checks if a swap between j1 and j2 is valid in terms of demand and if it improves the current solution
        """
        # Check if swap is possible in terms of demand and stocks
        product1 = self.seq[j1]
        product2 = self.seq[j2]

        for j in range(j1, j2):
            if self.instance.order(j,product1) :
                if self.stocks[product1][j] -1 <= 0 :
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

    def test_compute_cost(self, test_seq) :
        # Compute storage cost
        sum_date = 0
        for c in range(self.C):
            cnt = 0
            for j in range(self.J):
                if test_seq[j] == c:
                    cnt += 1
                if self.instance.order(j,c):
                    cnt -= 1
                sum_date += cnt
        # Compute transition costs
        sum_tran = 0
        last_c = -1
        for j in range(self.J):
            c = test_seq[j]
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
    # solver.hill_climbing_2_opt()
    solver.or_k_opt(k=3)
    
    sol = Solution(solver.best_seq)

    return sol

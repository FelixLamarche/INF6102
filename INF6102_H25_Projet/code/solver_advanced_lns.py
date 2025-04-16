import copy
import random
import sys
import time
from eternity_puzzle import EternityPuzzle
from solver_heuristic import solve_heuristic
from utils import *


class AdvancedSolver:
    def __init__(self, eternity_puzzle: EternityPuzzle, initial_solution: list[tuple[int, int, int, int]]):
        self.eternity_puzzle = eternity_puzzle
        self.board_size = eternity_puzzle.board_size

        self.solution = initial_solution
        self.n_conflicts = eternity_puzzle.get_total_n_conflict(self.solution)

        self.pieces_repaired = []
        self.positions_repaired = []

        self.prev_solution = None

    def solve_lns(self, percent_destroy=0.15) :
        """
        Solves the problem using LNS method
        """
        self.destroy_solution_conflicts_only(percent_destroy)
        if self.positions_repaired != [] :
            self.repair_solution_heuristic()
        self.n_conflicts = self.eternity_puzzle.get_total_n_conflict(self.solution)

    def destroy_solution_conflicts_only(self, percent_destroy=0.15) :
        """
        Destroy a part of the current solution
        Only destroy conflict pieces
        """
        self.prev_solution = copy.deepcopy(self.solution)
        conflict_pieces_and_positions = self.get_conflict_pieces()
        # Select pieces we will destroy
        nb_pieces_to_destroy = min(len(conflict_pieces_and_positions), round(percent_destroy * self.board_size**2))
        conflicts_selected = random.sample(conflict_pieces_and_positions, nb_pieces_to_destroy)
        self.pieces_repaired = ([conflicts_selected[i][0] for i in range(len(conflicts_selected))])
        self.positions_repaired = [(conflicts_selected[i][1], conflicts_selected[i][2]) for i in range(len(conflicts_selected))]
        # Remove them
        for x,y in self.positions_repaired :
            self.solution[get_idx(x, y, self.board_size)] = [-1,-1,-1,-1]

    def repair_solution_beam_search(self, beam_size=3) :
        """
        Repair the solution with a beam search
        """
        candidates = [{'moves': [], 'score' : self.n_conflicts}]
        random.shuffle(self.positions_repaired)
        for x,y in self.positions_repaired :
            new_candidates = []
            for candidate in candidates :
                used_pieces = set(id(original_piece) for original_piece, _, _ in candidate['moves'])
                available_pieces = [p for p in self.pieces_repaired if id(p) not in used_pieces]
                
                for piece in available_pieces :
                    piece_permutations = self.eternity_puzzle.generate_rotation(piece)

                    for permutation in piece_permutations:
                        temp_solution = self.apply_moves(candidate['moves'])
                        temp_solution[get_idx(x, y, self.board_size)] = permutation
                        n_conflicts = self.eternity_puzzle.get_total_n_conflict(temp_solution)

                        new_moves = candidate['moves'] + [(piece, (x, y), permutation)]
                        new_candidates.append({'moves' : new_moves, 'score' : n_conflicts})

            new_candidates.sort(key=lambda c: c['score'])
            candidates = new_candidates[:beam_size]

        best_candidate = candidates[0]['moves']
        self.solution = self.apply_moves(best_candidate)
        self.n_conflicts = self.eternity_puzzle.get_total_n_conflict(self.solution)

    def apply_moves(self, moves) :
        """
        Add a piece to the solution
        Used in repair_solution
        """
        solution = copy.deepcopy(self.solution)
        for _, (x,y), rotated_piece in moves:
            # if self.prev_solution[get_idx(x,y, self.board_size)] == rotated_piece :
            #     print("PIECE INCHANGEE")
            solution[get_idx(x,y, self.board_size)] = rotated_piece
        return solution 
    
    def repair_solution_heuristic(self):
        random.shuffle(self.positions_repaired)
        # self.positions_repaired = random.shuffle(self.positions_repaired)
        for x,y in self.positions_repaired :
            best_swaps = []
            best_conflict_count = 15
            for piece in self.pieces_repaired:
                piece_permutations = self.eternity_puzzle.generate_rotation(piece)
                for permutation in piece_permutations:
                    n_conflict = self.get_conflict_count(self.solution, x, y, permutation)
                    if n_conflict < best_conflict_count:
                        best_swaps = [(piece, permutation)]
                        best_conflict_count = n_conflict
                    elif n_conflict == best_conflict_count:
                        best_swaps.append((piece, permutation))
                
            best_swap = random.choice(best_swaps)
            self.solution[get_idx(x, y, self.board_size)] = best_swap[1]
            self.pieces_repaired.remove(best_swap[0])
            self.n_conflicts = self.eternity_puzzle.get_total_n_conflict(self.solution)
    
    def get_conflict_pieces(self) :
        conflict_pieces = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                piece = get_piece(self.solution, x, y, self.board_size)
                piece_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece, self.board_size)
                if piece_conflict_count > 0 :
                    conflict_pieces.append((piece, x, y))
        return conflict_pieces
    
    def get_conflict_count(self, solution, x, y, piece):
            n_conflict = 0

            if is_corner(x, y, self.board_size):
                if not is_corner_piece(piece):
                    n_conflict += 10 # non-corner pieces should not be in the corner
            elif is_edge(x, y, self.board_size) and not is_edge_piece(piece):
                    n_conflict += 10 #non-edge pieces should not be in the edge

            north_piece = get_piece(solution, x, y + 1, self.board_size)
            south_piece = get_piece(solution, x, y - 1, self.board_size)
            west_piece = get_piece(solution, x - 1, y, self.board_size)
            east_piece = get_piece(solution, x + 1, y, self.board_size)
            if north_piece != [-1,-1,-1,-1] and piece[NORTH] != north_piece[SOUTH]:
                n_conflict += 1
            if south_piece != [-1,-1,-1,-1] and piece[SOUTH] != south_piece[NORTH]:
                n_conflict += 1
            if west_piece != [-1,-1,-1,-1] and piece[WEST] != west_piece[EAST]:
                n_conflict += 1
            if east_piece != [-1,-1,-1,-1] and piece[EAST] != east_piece[WEST]:
                n_conflict += 1
            return n_conflict
    

def get_initial_solution_semi_random(eternity_puzzle: EternityPuzzle) -> list[tuple[int, int, int, int]]:
    """
    Returns an initial solution with pieces randomly placed, but with the edge, corner and internal pieces in their correct placement on the board in a random rotation
    """
    piece_list = copy.deepcopy(eternity_puzzle.piece_list)
    corner_pieces = [piece for piece in piece_list if is_corner_piece(piece)]
    edge_pieces = [piece for piece in piece_list if is_edge_piece(piece)]
    internal_pieces = [piece for piece in piece_list if not is_corner_piece(piece) and not is_edge_piece(piece)]
    random.shuffle(corner_pieces)
    random.shuffle(edge_pieces)
    random.shuffle(internal_pieces)
    
    corner_idx = 0
    edge_idx = 0
    internal_idx = 0
    for x in range(eternity_puzzle.board_size):
        for y in range(eternity_puzzle.board_size):
            idx = get_idx(x, y, eternity_puzzle.board_size)
            if is_corner(x, y, eternity_puzzle.board_size):
                piece_list[idx] = corner_pieces[corner_idx]
                corner_idx += 1
            elif is_edge(x, y, eternity_puzzle.board_size):
                piece_list[idx] = edge_pieces[edge_idx]
                edge_idx += 1
            else:
                piece_list[idx] = internal_pieces[internal_idx]
                internal_idx += 1
            piece_list[idx] = random.choice(eternity_puzzle.generate_rotation(piece_list[idx]))

    return piece_list

def get_initial_solution(eternity_puzzle: EternityPuzzle) -> list[tuple[int, int, int, int]]:
    """
    Returns an initial solution with the same layout as the input
    """
    return copy.deepcopy(eternity_puzzle.piece_list)

def get_initial_solution_heuristic(eternity_puzzle: EternityPuzzle) -> list[tuple[int, int, int, int]]:
    return solve_heuristic(eternity_puzzle)[0]

def shuffle_solution(solution: list[tuple[int, int, int, int]], nb_pieces_to_shuffle: int) -> list[tuple[int, int, int, int]]:
    """
    Copies and shuffles the solution by randomly swapping a specific amount of pieces between each other
    In essence a perturbation method for an iterative local search
    """
    board_size = int(len(solution) ** 0.5)
    positions = [(x, y) for x in range(board_size) for y in range(board_size)]
    positions_to_shuffle = random.choices(positions, k=nb_pieces_to_shuffle)
    new_solution = copy.deepcopy(solution)
    
    for i in range(nb_pieces_to_shuffle):
        x1, y1 = positions_to_shuffle[i]
        x2, y2 = positions_to_shuffle[(i + 1) % nb_pieces_to_shuffle]
        idx1 = get_idx(x1, y1, board_size)
        idx2 = get_idx(x2, y2, board_size)
        temp = new_solution[idx1]
        new_solution[idx1] = new_solution[idx2]
        new_solution[idx2] = temp
    
    return new_solution


def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    TIME_SEARCH_SEC = 3600 # Will be 1 hour for the final version
    TIME_SEARCH_MARGIN = 5 # PUT This to 5 seconds to avoid timing out in the final version

    NB_ITERATIONS_NO_IMPROVEMENT = 100

    best_solution = None # Best solution found so far
    best_conflict_count = 999
    best_solution_cur_search = None # Best solution found in the current search
    best_conflict_count_cur_search = 999

    time_before = time.time()
    initial_solution = None
    iteration_count = 0
    iteration_without_improvement = 0
    total_iterations = 0

    while time.time() - time_before < TIME_SEARCH_SEC - TIME_SEARCH_MARGIN :

        time_start_iter = time.time()
        time_to_search = TIME_SEARCH_SEC - (time.time() - time_before) - TIME_SEARCH_MARGIN

        if iteration_without_improvement >= NB_ITERATIONS_NO_IMPROVEMENT:
            # Chance to either restart a new search, or resume from the best solution found so far
            
            if random.uniform(0.0, 1.0) < 0.5:
                best_solution_cur_search = None 
                best_conflict_count_cur_search = 999
            else:
                best_solution_cur_search = copy.deepcopy(best_solution)
                best_conflict_count_cur_search = best_conflict_count
            iteration_without_improvement = 0
            percent_destroy = 0.3

        initial_solution = best_solution_cur_search
        nb_pieces_to_shuffle = 0
        ratio_of_conflicts = 0
        if initial_solution is None: # If there isn't an initial solution
            initial_solution = get_initial_solution_heuristic(eternity_puzzle)
            percent_destroy = 0.3
        
        solver = AdvancedSolver(eternity_puzzle, initial_solution)

        # Restart search if no improvement were made
        solver.solve_lns(percent_destroy)

        iteration_count += 1
        iteration_without_improvement += 1

        if solver.n_conflicts < best_conflict_count_cur_search:
            best_conflict_count_cur_search = solver.n_conflicts
            best_solution_cur_search = copy.deepcopy(solver.solution)
            iteration_without_improvement = 0
            percent_destroy = max(0.05, percent_destroy*0.8)

            # print("NewLocalBest: Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter))

        #print("Conflicts: ", solver.n_conflicts)
        if solver.n_conflicts < best_conflict_count:
            best_conflict_count = solver.n_conflicts
            best_solution = copy.deepcopy(solver.solution)
            print("NewGlobalBest: Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter), "percent_destroy : ", percent_destroy)

        if best_conflict_count == 0:
            # print('COMPLET')
            break
    return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)

# def solve_advanced(eternity_puzzle):
#     """
#     Your solver for the problem
#     :param eternity_puzzle: object describing the input
#     :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
#         cost is the cost of the solution
#     """
#     TIME_SEARCH_SEC = 30
#     TIME_MARGIN_SEC = 15

#     best_conflict_count = 500
#     best_solution = None

    # time_before = time.time()
    # iteration_count = 0
    
#     # while time.time() - time_before + TIME_MARGIN_SEC < TIME_SEARCH_SEC:
#     #     iteration_count += 1

#     #     # Set a seed for the random number generator to get different results each time
#     #     seed = random.randint(1, sys.maxsize)
#     #     random.seed(seed)
#     #     print("Seed: ", seed)

#     #     init_solution = get_initial_solution_semi_random(eternity_puzzle)
#     #     nb_pieces_to_shuffle = int(len(init_solution) * 0.1)
#     #     init_solution = shuffle_solution(init_solution, nb_pieces_to_shuffle)
#     #     print("Shuffled ", nb_pieces_to_shuffle, " pieces")

#     #     solver = AdvancedSolver(eternity_puzzle, init_solution)
#     #     # solver.solve_local_search()
#     #     if best_conflict_count == 0:
#     #         break
#     #     solver.solve_lns()
#     #     if solver.n_conflicts < best_conflict_count:
#     #         print(f'Previous best_conflit_count : {best_conflict_count}, new : {solver.n_conflicts}')
#     #         print(solver.solution)
#     #         best_conflict_count = solver.n_conflicts
#     #         best_solution = solver.solution
#     #     if best_conflict_count == 0:
#     #         break
#     #     # if best_conflict_count == 0:
#     #     #     break
#     #     print("Iteration ", iteration_count, " conflicts: ", solver.n_conflicts)
    
    # initial_solution = get_initial_solution_heuristic(eternity_puzzle)
    # percentage_of_shuffle = random.uniform(0.01, 0.80)
    # nb_pieces_to_shuffle = int(len(initial_solution) * percentage_of_shuffle)
    # # nb_pieces_to_shuffle = int(len(initial_solution) * 0.1)
    # initial_solution = shuffle_solution(initial_solution, nb_pieces_to_shuffle)
    # print("Shuffled ", nb_pieces_to_shuffle, " pieces")
    # solver = AdvancedSolver(eternity_puzzle, initial_solution)
    # solver.solve_lns()
    # best_solution = solver.solution
    # best_conflict_count = eternity_puzzle.get_total_n_conflict(best_solution)
    # print(f"NewSolution: {best_conflict_count} conflicts")

    # return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)

# def solve_advanced(eternity_puzzle):
#     board_size = eternity_puzzle.board_size
#     time_limit = 60  # seconds
#     total_time_limit = 3500  # 1 hour
#     start_time = time.time()

#     initial_solution = get_initial_solution_semi_random(eternity_puzzle)
#     best_solution = initial_solution.copy()
#     best_conflict_count = 500
#     conflict_history = []

#     iterations = 0
#     conflict_history.append((iterations, best_conflict_count))

#     while time.time() - start_time < total_time_limit:
#         iterations += 1

#         # Run your local solver
#         solver = AdvancedSolver(eternity_puzzle, initial_solution)
#         solver.solve_local_search(time_limit)


#         if solver.n_conflicts < best_conflict_count:
#             best_conflict_count = solver.n_conflicts
#             best_solution = solver.solution
#             conflict_history.append((iterations, best_conflict_count))
#             print(f"Iteration {iterations}: New best conflicts = {best_conflict_count}")

        

#         if best_conflict_count == 0:
#             print("Perfect solution found!")
#             break

#         # --- Decide what to do next ---
#         if iterations % 3 == 0:
#             # Every 3 iterations, apply LNS
#             # n_pieces_to_remove = random.randint(5, 20)  # tune this number
#             solver.destroy_solution(percent_destroy=0.15)
#             solver.repair_solution_heuristic()
#             # print(f"LNS applied: Removed and repaired pieces.")
#         else:
#             # Otherwise, random shuffle restart
#             percentage_of_shuffle = random.uniform(0.01, 0.80)
#             nb_pieces_to_shuffle = int(len(initial_solution) * percentage_of_shuffle)
#             initial_solution = shuffle_solution(initial_solution, nb_pieces_to_shuffle)
#             # print("Shuffled ", nb_pieces_to_shuffle, " pieces")

#     return best_solution, eternity_puzzle.get_total_n_conflict(best_solution), conflict_history


# def solve_local_search_only(eternity_puzzle):
#     board_size = eternity_puzzle.board_size
#     time_limit = 60  # seconds
#     total_time_limit = 3500  # 1 hour
#     start_time = time.time()

#     initial_solution = get_initial_solution_semi_random(eternity_puzzle)
#     best_solution = initial_solution.copy()
#     best_conflict_count = 500
#     conflict_history = []

#     iterations = 0
#     conflict_history.append((iterations, best_conflict_count))

#     while time.time() - start_time < total_time_limit:
#         iterations += 1

#         # Run your local solver
#         solver = AdvancedSolver(eternity_puzzle, initial_solution)
#         solver.solve_local_search(time_limit)


#         if solver.n_conflicts < best_conflict_count:
#             best_conflict_count = solver.n_conflicts
#             best_solution = solver.solution
#             print(f"Iteration {iterations}: New best conflicts = {best_conflict_count}")
#             conflict_history.append((iterations, best_conflict_count))


#         if best_conflict_count == 0:
#             print("Perfect solution found!")
#             break

#         percentage_of_shuffle = random.uniform(0.01, 0.80)
#         nb_pieces_to_shuffle = int(len(initial_solution) * percentage_of_shuffle)
#         initial_solution = shuffle_solution(initial_solution, nb_pieces_to_shuffle)
#         # print("Shuffled ", nb_pieces_to_shuffle, " pieces")

#     return best_solution, eternity_puzzle.get_total_n_conflict(best_solution), conflict_history


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

    def solve_lns(self, percent_destroy=0.5) :
        """
        Solves the problem using LNS method
        """
        
        # print(f"CONFLICTS BEFORE REPAIR : {self.n_conflicts}, solution : {self.solution}")
        self.destroy_solution_conflicts_only(percent_destroy)
        # print(f"POSITIONS DESTROYED : {self.positions_repaired}, PIECES : {self.pieces_repaired}")
        # self.repair_solution_heuristic()
        self.repair_solution_beam_search(beam_size=3)
        # print(f"CONFLICTS AFTER REPAIR : {self.n_conflicts}, solution : {self.solution}")

    def destroy_solution_conflicts_only(self, percent_destroy=0.15) :
        """
        Destroy a part of the current solution
        Only destroy conflict pieces
        """
        conflict_pieces_and_positions = self.get_conflict_pieces()
        # Select pieces we will destroy
        nb_pieces_to_destroy = min(len(conflict_pieces_and_positions), round(percent_destroy * self.board_size**2))
        conflicts_selected = random.sample(conflict_pieces_and_positions, nb_pieces_to_destroy)
        self.pieces_repaired = ([conflicts_selected[i][0] for i in range(len(conflicts_selected))])
        self.positions_repaired = [(conflicts_selected[i][1], conflicts_selected[i][2]) for i in range(len(conflicts_selected))]
        # print(self.pieces_repaired, self.positions_repaired)
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
                    # We check if any permutation of the piece has already been used

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

    def apply_moves(self, moves) :
        """
        Add a piece to the solution
        Used in repair_solution
        """
        solution = copy.deepcopy(self.solution)
        for _, (x,y), rotated_piece in moves:
            solution[get_idx(x,y, self.board_size)] = rotated_piece
        return solution 

    def solve_local_search(self, time_to_search_sec: float = 60):
        """
        Solves the problem using a local search algorithm, until a local minimum is reached, by doing LKH swaps
        """
        # Create a neighborhood from a 2-opt, where we swap two pieces (can be a piece with another rotation of itself)
        # We select the best neighbor (the one with the lowest cost) and keep going until we reach a local minimum
        # We repeat this process from different random starts and keep the best solution found
        MIN_CONFLICTS_SOLVED = 1

        time_limit = time.time() + time_to_search_sec
        prev_conflict_count = 9999999
        #print("Conflicts start: ", self.n_conflicts)
        while prev_conflict_count - self.n_conflicts >= MIN_CONFLICTS_SOLVED and time.time() < time_limit:
            prev_conflict_count = self.n_conflicts
            #print("Conflicts: ", self.n_conflicts)
            board_positions = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
            random.shuffle(board_positions)
            for x1, y1 in board_positions:
                if time.time() > time_limit:
                    break
                piece1 = get_piece(self.solution, x1, y1, self.board_size)
                self.do_lkh_swap(x1, y1, piece1)
  
    def swap_pieces(self, x1: int, y1: int, new_piece1: tuple, x2: int, y2: int, new_piece2: tuple):
        """
        Swaps two pieces (in their target rotations) in the solution, and updates the conflict count
        """
        # Get the previous conflicts from the original rotation of the pieces
        prev_piece1 = get_piece(self.solution, x1, y1, self.board_size)
        prev_piece2 = get_piece(self.solution, x2, y2, self.board_size)
        prev_n_conflicts = get_conflict_count_for_piece(self.solution, x1, y1, prev_piece1, self.board_size)

        are_pieces_same_position = x1 == x2 and y1 == y2
        if not are_pieces_same_position:
            prev_n_conflicts += get_conflict_count_for_piece(self.solution, x2, y2, prev_piece2, self.board_size)
        # Dont count twice the same conflict
        if has_conflict_with_adjacent_piece(x1, y1, prev_piece1, x2, y2, prev_piece2):
            prev_n_conflicts -= 1

        self.solution[get_idx(x1, y1, self.board_size)] = new_piece2
        if not are_pieces_same_position:
            self.solution[get_idx(x2, y2, self.board_size)] = new_piece1

        new_n_conflicts = get_conflict_count_for_piece(self.solution, x1, y1, new_piece2, self.board_size)
        if not are_pieces_same_position:
            new_n_conflicts += get_conflict_count_for_piece(self.solution, x2, y2, new_piece1, self.board_size)
        # Dont count twice the same conflict
        if has_conflict_with_adjacent_piece(x2, y2, new_piece1, x1, y1, new_piece2):
            new_n_conflicts -= 1

        # Update the running conflict count
        self.n_conflicts += new_n_conflicts - prev_n_conflicts

    def do_lkh_swap(self, x: int, y: int, piece1: tuple, repair=False) -> int:
        """
        Does a LKH swap, where we continuously swap pieces until no more beneficial swaps are found
        Returns the number of swaps done
        """
        nb_swaps = 0

        x_to_swap = x
        y_to_swap = y
        piece_to_swap = piece1
        while piece_to_swap is not None:
            piece2_to_swap, piece2_coord, piece_to_swap_rotated = self.get_best_swap(x_to_swap, y_to_swap, piece_to_swap, repair)
            if piece2_to_swap is None:
                break

            self.swap_pieces(x_to_swap, y_to_swap, piece_to_swap_rotated, piece2_coord[0], piece2_coord[1], piece2_to_swap)
            piece_to_swap = piece2_to_swap

            nb_swaps += 1

        return nb_swaps
                     
    def get_best_swap(self, x: int, y: int, piece1: tuple, repair=False) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple]:
        """
        Returns the piece to swap with and its coordinates, and the correct piece rotation, if a beneficial swap is found
        """
        class SwapInfo:
            # This class is used to keep track of the best swap found so far
            def __init__(self, piece1_rotation: tuple, piece2_coord: tuple[int, int], piece2_to_swap: tuple, swap_conflict_delta: int):
                self.piece1_rotation = piece1_rotation
                self.piece2_coord = piece2_coord
                self.piece2_to_swap = piece2_to_swap
                self.swap_conflict_delta = swap_conflict_delta

        if repair :
            positions_to_test = self.positions_repaired
        else :
            positions_to_test = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]

        prev_piece1_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece1, self.board_size)
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)
        # Keep track of the best swap so far
        best_swaps = [SwapInfo(None, (-1, -1), None, 0)]
        # Check every piece and rotation as a possible swap
        # for x2 in range(self.board_size):
        #     for y2 in range(self.board_size):
        for x2, y2 in positions_to_test :
            piece2 = get_piece(self.solution, x2, y2, self.board_size)
            prev_piece2_conflict_count = get_conflict_count_for_piece(self.solution, x2, y2, piece2, self.board_size)
            # Dont count twice the same conflict
            if has_conflict_with_adjacent_piece(x, y, piece1, x2, y2, piece2):
                prev_piece2_conflict_count -= 1

            # Evaluate the swap 
            best_piece1_rotation_attempt = None
            best_piece1_rotation_conflict_count = 999999
            best_piece2_rotation = None
            best_piece2_conflict_count = 999999
            if are_coords_adjacent(x, y, x2, y2): # If they are adjacent, they influence each other, so we calculate the conflicts differently
                # Check every possible rotation of the pieces by swapping them and evaluating the conflicts
                piece1_idx = get_idx(x, y, self.board_size)
                piece2_idx = get_idx(x2, y2, self.board_size)
                for piece1_rotation in piece1_rotations:
                    for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                        self.solution[piece2_idx] = piece1_rotation
                        self.solution[piece1_idx] = piece2_rotation

                        piece1_rotation_conflict_count = get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation, self.board_size)
                        piece2_rotation_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece2_rotation, self.board_size)
                        # Dont count twice the same conflict
                        if has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
                            piece2_rotation_conflict_count -= 1

                        if piece1_rotation_conflict_count + piece2_rotation_conflict_count < best_piece1_rotation_conflict_count + best_piece2_conflict_count:
                            best_piece1_rotation_conflict_count = piece1_rotation_conflict_count
                            best_piece1_rotation_attempt = piece1_rotation
                            best_piece2_conflict_count = piece2_rotation_conflict_count
                            best_piece2_rotation = piece2_rotation
                # Reset the pieces to their original state
                self.solution[piece1_idx] = piece1
                self.solution[piece2_idx] = piece2
            else: # We can evaluate the swap without swapping if they are not adjacent and do not have conflicts between each other, and check less possibilities (16 to 8)
                # Get the best rotation for the original piece in piece2's place
                for piece1_rotation in piece1_rotations:
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation, self.board_size)
                    if swap_conflict_count < best_piece1_rotation_conflict_count:
                        best_piece1_rotation_conflict_count = swap_conflict_count
                        best_piece1_rotation_attempt = piece1_rotation

                # Get the best rotation for the piece2 in the original piece's place
                for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece2_rotation, self.board_size)
                    # Dont count twice the same conflict
                    if has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
                        swap_conflict_count -= 1
                    if swap_conflict_count < best_piece2_conflict_count:
                        best_piece2_rotation = piece2_rotation
                        best_piece2_conflict_count = swap_conflict_count

            # Check if the swap is the beneficial and the best
            delta_conflict_count = best_piece2_conflict_count + best_piece1_rotation_conflict_count - prev_piece1_conflict_count - prev_piece2_conflict_count

            if delta_conflict_count < best_swaps[0].swap_conflict_delta:
                # If the swap is better we replace the best swaps found so far
                best_swaps = [SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, delta_conflict_count)]
            elif delta_conflict_count == best_swaps[0].swap_conflict_delta:
                # If the swap is equal to the best, we add it to the list of best swaps
                best_swaps.append(SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, delta_conflict_count))
    
        chosen_swap = random.choice(best_swaps) # Randomly select one of the best swaps found
        return (chosen_swap.piece2_to_swap, chosen_swap.piece2_coord, chosen_swap.piece1_rotation)

    def get_conflict_pieces(self) :
        conflict_pieces = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                piece = get_piece(self.solution, x, y, self.board_size)
                piece_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece, self.board_size)
                if piece_conflict_count > 0 :
                    conflict_pieces.append((piece, x, y))
        return conflict_pieces

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
    TIME_SEARCH_SEC = 600 # Will be 1 hour for the final version
    TIME_SEARCH_MARGIN = 5 # PUT This to 5 seconds to avoid timing out in the final version

    NB_ITERATIONS_NO_IMPROVEMENT = 100

    best_solution = None # Best solution found so far
    best_conflict_count = 999
    best_solution_cur_search = None # Best solution found in the current search
    best_conflict_count_cur_search = 999
    percent_destroy = 0.4

    time_before = time.time()
    initial_solution = None
    iteration_count = 0
    iteration_without_improvement = 0
    while time.time() - time_before < TIME_SEARCH_SEC - TIME_SEARCH_MARGIN:
        time_start_iter = time.time()
        time_to_search = TIME_SEARCH_SEC - (time.time() - time_before) - TIME_SEARCH_MARGIN

        # Set a random seed
        seed = random.randint(1, sys.maxsize)
        random.seed(seed)
        #print("Seed: ", seed)

        # Restart search if no improvement were made
        if iteration_without_improvement >= NB_ITERATIONS_NO_IMPROVEMENT:
            best_solution_cur_search = None 
            best_conflict_count_cur_search = 999
            percent_destroy = 0.4

        initial_solution = best_solution_cur_search
        nb_pieces_to_shuffle = 0
        if initial_solution is None:
            initial_solution = get_initial_solution_heuristic(eternity_puzzle)
        else:
            percentage_of_shuffle = random.uniform(0.01, 0.80)
            nb_pieces_to_shuffle = int(len(initial_solution) * percentage_of_shuffle)
            initial_solution = shuffle_solution(initial_solution, nb_pieces_to_shuffle)
            #print("Shuffled ", nb_pieces_to_shuffle, " pieces")

        solver = AdvancedSolver(eternity_puzzle, initial_solution)
        # solver.solve(time_to_search)
        # solver.solve_local_search(time_to_search)
        solver.solve_lns(percent_destroy)

        iteration_count += 1
        iteration_without_improvement += 1

        if solver.n_conflicts < best_conflict_count_cur_search:
            best_conflict_count_cur_search = solver.n_conflicts
            best_solution_cur_search = solver.solution
            percent_destroy = min(percent_destroy*0.8, 0.1)
            iteration_without_improvement = 0
            # print("NewLocalBest : Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter), " nb pieces shuffled: ", nb_pieces_to_shuffle)

        #print("Conflicts: ", solver.n_conflicts)
        if solver.n_conflicts < best_conflict_count:
            best_conflict_count = solver.n_conflicts
            best_solution = solver.solution
            print("NewGlobalBest: Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter), " nb pieces shuffled: ", nb_pieces_to_shuffle)


        if best_conflict_count == 0:
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


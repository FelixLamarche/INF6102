import copy
import random
import sys
import time
from eternity_puzzle import EternityPuzzle
from solver_heuristic import solve_heuristic
from utils import *

class LocalSolver:
    def __init__(self, eternity_puzzle: EternityPuzzle, initial_solution: list[tuple[int, int, int, int]]):
        self.eternity_puzzle = eternity_puzzle
        self.board_size = eternity_puzzle.board_size

        # Set the initial solution
        self.solution = initial_solution
        self.n_conflicts = self.eternity_puzzle.get_total_n_conflict(initial_solution)

    def solve(self):
        """
        Solves the problem using a local search algorithm, until a local minimum is reached
        """
        # Create a neighborhood from a 2-opt, where we swap two pieces (can be a piece with another rotation of itself)
        # We select the best neighbor (the one with the lowest cost) and keep going until we reach a local minimum
        # We repeat this process from different random starts and keep the best solution found
        MIN_CONFLICTS_SOLVED = 1

        prev_conflict_count = 9999999
        print("Conflicts start: ", self.n_conflicts)
        while prev_conflict_count - self.n_conflicts >= MIN_CONFLICTS_SOLVED:
            prev_conflict_count = self.n_conflicts
            print("Conflicts: ", self.n_conflicts)
            board_positions = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
            random.shuffle(board_positions)

            for x1, y1 in board_positions:
                piece1 = get_piece(self.solution, x1, y1, self.board_size)
                piece2_to_swap, piece2_to_swap_coord, piece1_rotated = self.get_best_swap(x1, y1, piece1)
                if piece2_to_swap is not None:
                    self.swap_pieces(x1, y1, piece1_rotated, piece2_to_swap_coord[0], piece2_to_swap_coord[1], piece2_to_swap)
    
    def swap_pieces(self, x1: int, y1: int, new_piece1: tuple, x2: int, y2: int, new_piece2: tuple):
        """
        Swaps two pieces (in their target rotations) in the solution, and updates the conflict count
        """
        # Get the previous conflicts from the original rotation of the pieces
        prev_piece1 = get_piece(self.solution, x1, y1, self.board_size)
        prev_piece2 = get_piece(self.solution, x2, y2, self.board_size)
        prev_n_conflicts = get_conflict_count_for_piece(self.solution, x1, y1, prev_piece1, self.board_size)
        if x1 != x2 or y1 != y2:
            prev_n_conflicts += get_conflict_count_for_piece(self.solution, x2, y2, prev_piece2, self.board_size)
        # Dont count twice the same conflict
        if has_conflict_with_adjacent_piece(x1, y1, prev_piece1, x2, y2, prev_piece2):
            prev_n_conflicts -= 1

        self.solution[get_idx(x1, y1, self.board_size)] = new_piece2
        self.solution[get_idx(x2, y2, self.board_size)] = new_piece1

        new_n_conflicts = get_conflict_count_for_piece(self.solution, x1, y1, new_piece2, self.board_size)
        if x1 != x2 or y1 != y2:
            new_n_conflicts += get_conflict_count_for_piece(self.solution, x2, y2, new_piece1, self.board_size)
        # Dont count twice the same conflict
        if has_conflict_with_adjacent_piece(x2, y2, new_piece1, x1, y1, new_piece2):
            new_n_conflicts -= 1

        # Update the running conflict count
        self.n_conflicts += new_n_conflicts - prev_n_conflicts
                            
    def get_best_swap(self, x: int, y: int, piece1: tuple) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple]:
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

        prev_piece1_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece1, self.board_size)
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)
        # Keep track of the best swap so far
        best_swaps = [SwapInfo(None, (-1, -1), None, 0)]
        # Check every piece and rotation as a possible swap
        for x2 in range(self.board_size):
            for y2 in range(self.board_size):
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


def get_initial_solution_pseudo_random(eternity_puzzle: EternityPuzzle) -> list[tuple[int, int, int, int]]:
    """
    Returns an initial solution with pieces randomly placed on the board in a random rotation
    """
    piece_list = copy.deepcopy(eternity_puzzle.piece_list)
    random.shuffle(piece_list)
    for i in range(len(piece_list)):
        rotations = eternity_puzzle.generate_rotation(piece_list[i])
        piece_list[i] = random.choice(rotations)

    return piece_list

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


def solve_local_search(eternity_puzzle: EternityPuzzle) -> tuple[list[tuple[int, int, int, int]], int]:
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    TIME_SEARCH_SEC = 180
    TIME_MARGIN_SEC = 15

    best_conflict_count = 500
    best_solution = None

    time_before = time.time()
    iteration_count = 0
    
    while time.time() - time_before + TIME_MARGIN_SEC < TIME_SEARCH_SEC:
        iteration_count += 1

        # Set a seed for the random number generator to get different results each time
        seed = random.randint(1, sys.maxsize)
        random.seed(seed)
        print("Seed: ", seed)

        init_solution = get_initial_solution_heuristic(eternity_puzzle)
        solver = LocalSolver(eternity_puzzle, init_solution)
        solver.solve()
        if solver.n_conflicts < best_conflict_count:
            best_conflict_count = solver.n_conflicts
            best_solution = solver.solution
        if best_conflict_count == 0:
            break
        print("Iteration ", iteration_count, " conflicts: ", solver.n_conflicts)

    return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)

import copy
import random
import sys
import time
from eternity_puzzle import EternityPuzzle
from utils import *

class LocalSolver:
    def __init__(self, eternity_puzzle: EternityPuzzle):
        self.eternity_puzzle = eternity_puzzle
        self.board_size = eternity_puzzle.board_size

        # Get an initial solution (random)
        seed = random.randint(1, sys.maxsize)
        random.seed(seed)
        print("Seed: ", seed)

        self.solution = self.get_initial_solution_semi_random()

        self.n_conflicts = eternity_puzzle.get_total_n_conflict(self.solution)

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

            for x1 in range(self.board_size):
                for y1 in range(self.board_size):
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
        prev_n_conflicts = self.get_conflict_count_for_piece(self.solution, x1, y1, prev_piece1)
        if x1 != x2 or y1 != y2:
            prev_n_conflicts += self.get_conflict_count_for_piece(self.solution, x2, y2, prev_piece2)
        # Dont count twice the same conflict
        if self.has_conflict_with_adjacent_piece(x1, y1, prev_piece1, x2, y2, prev_piece2):
            prev_n_conflicts -= 1

        self.solution[get_idx(x1, y1, self.board_size)] = new_piece2
        self.solution[get_idx(x2, y2, self.board_size)] = new_piece1

        new_n_conflicts = self.get_conflict_count_for_piece(self.solution, x1, y1, new_piece2)
        if x1 != x2 or y1 != y2:
            new_n_conflicts += self.get_conflict_count_for_piece(self.solution, x2, y2, new_piece1)
        # Dont count twice the same conflict
        if self.has_conflict_with_adjacent_piece(x2, y2, new_piece1, x1, y1, new_piece2):
            new_n_conflicts -= 1

        # Update the running conflict count
        self.n_conflicts += new_n_conflicts - prev_n_conflicts
                            
    def get_best_swap(self, x: int, y: int, piece1: tuple) -> tuple:
        """
        Returns the piece to swap with and its coordinates, and the correct piece rotation, if a beneficial swap is found
        """
        prev_piece1_conflict_count = self.get_conflict_count_for_piece(self.solution, x, y, piece1)
        
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)

        best_piece1_rotation = None
        best_piece2_coord = (-1, -1)
        best_piece2_to_swap = None
        best_swap_conflict_delta = 0
        # Check every piece and rotation as a possible swap
        for x2 in range(self.board_size):
            for y2 in range(self.board_size):
                piece2 = get_piece(self.solution, x2, y2, self.board_size)
                prev_piece2_conflict_count = self.get_conflict_count_for_piece(self.solution, x2, y2, piece2)
                # Dont count twice the same conflict
                if self.has_conflict_with_adjacent_piece(x, y, piece1, x2, y2, piece2):
                    prev_piece2_conflict_count -= 1

                best_piece1_rotation_attempt = None
                best_piece1_rotation_conflict_count = 999999
                best_piece2_rotation = None
                best_piece2_conflict_count = 999999
                if are_coords_adjacent(x, y, x2, y2):
                    # Check every possible rotation of the pieces by swapping them and evaluating the conflicts
                    piece1_idx = get_idx(x, y, self.board_size)
                    piece2_idx = get_idx(x2, y2, self.board_size)
                    for piece1_rotation in piece1_rotations:
                        for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                            self.solution[piece2_idx] = piece1_rotation
                            self.solution[piece1_idx] = piece2_rotation

                            piece1_rotation_conflict_count = self.get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation)
                            piece2_rotation_conflict_count = self.get_conflict_count_for_piece(self.solution, x, y, piece2_rotation)
                            # Dont count twice the same conflict
                            if self.has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
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
                        swap_conflict_count = self.get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation)
                        if swap_conflict_count < best_piece1_rotation_conflict_count:
                            best_piece1_rotation_conflict_count = swap_conflict_count
                            best_piece1_rotation_attempt = piece1_rotation

                    # Get the best rotation for the piece2 in the original piece's place
                    for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                        swap_conflict_count = self.get_conflict_count_for_piece(self.solution, x, y, piece2_rotation)
                        # Dont count twice the same conflict
                        if self.has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
                            swap_conflict_count -= 1
                        if swap_conflict_count < best_piece2_conflict_count:
                            best_piece2_rotation = piece2_rotation
                            best_piece2_conflict_count = swap_conflict_count

                # Check if the swap is the beneficial and the best
                delta_conflict_count = best_piece2_conflict_count + best_piece1_rotation_conflict_count - prev_piece1_conflict_count - prev_piece2_conflict_count

                if delta_conflict_count < best_swap_conflict_delta:
                    best_piece2_to_swap = best_piece2_rotation
                    best_piece2_coord = (x2, y2)
                    best_piece1_rotation = best_piece1_rotation_attempt
                    best_swap_conflict_delta = delta_conflict_count
        
        return (best_piece2_to_swap, best_piece2_coord, best_piece1_rotation)

    def has_conflict_with_adjacent_piece(self, x1:int, y1:int, piece1:tuple, x2:int, y2:int, piece2:tuple) -> bool:
        """
        Returns True if two pieces have a conflict between each other
        """
        if not are_coords_adjacent(x1, y1, x2, y2):
            return False

        if x1 + 1 == x2:
            return piece1[EAST] != piece2[WEST]
        if x1 - 1 == x2:
            return piece1[WEST] != piece2[EAST]
        if y1 + 1 == y2:
            return piece1[NORTH] != piece2[SOUTH]
        if y1 - 1 == y2:
            return piece1[SOUTH] != piece2[NORTH]
        return False

    def get_conflict_count_for_piece(self, solution: list, x: int, y: int, piece: tuple) -> int:
        """
        Returns the number of conflicts that a piece would have if placed at the position
        """
        n_conflict = 0
        # Check NORTH
        if piece[NORTH] != get_piece(solution, x, y + 1, self.board_size)[SOUTH]:
            n_conflict += 1
        if piece[SOUTH] != get_piece(solution, x, y - 1, self.board_size)[NORTH]:
            n_conflict += 1
        if piece[WEST] != get_piece(solution, x - 1, y, self.board_size)[EAST]:
            n_conflict += 1
        if piece[EAST] != get_piece(solution, x + 1, y, self.board_size)[WEST]:
            n_conflict += 1
        return n_conflict

    def get_initial_solution_semi_random(self) -> list[tuple[int, int, int, int]]:
        """
        Returns an initial solution with pieces randomly placed on the board in a random rotation
        """
        piece_list = copy.deepcopy(self.eternity_puzzle.piece_list)
        random.shuffle(piece_list)
        for i in range(len(piece_list)):
            rotations = self.eternity_puzzle.generate_rotation(piece_list[i])
            piece_list[i] = random.choice(rotations)

        return piece_list

    def get_initial_solution_semi_random(self) -> list[tuple[int, int, int, int]]:
        """
        Returns an initial solution with pieces randomly placed, but with the edge, corner and internal pieces in their correct placement on the board in a random rotation
        """
        piece_list = copy.deepcopy(self.eternity_puzzle.piece_list)
        corner_pieces = [piece for piece in piece_list if is_corner_piece(piece)]
        edge_pieces = [piece for piece in piece_list if is_edge_piece(piece)]
        internal_pieces = [piece for piece in piece_list if not is_corner_piece(piece) and not is_edge_piece(piece)]
        random.shuffle(corner_pieces)
        random.shuffle(edge_pieces)
        random.shuffle(internal_pieces)
        
        corner_idx = 0
        edge_idx = 0
        internal_idx = 0
        for x in range(self.board_size):
            for y in range(self.board_size):
                idx = get_idx(x, y, self.board_size)
                if is_corner(x, y, self.board_size):
                    piece_list[idx] = corner_pieces[corner_idx]
                    corner_idx += 1
                elif is_edge(x, y, self.board_size):
                    piece_list[idx] = edge_pieces[edge_idx]
                    edge_idx += 1
                else:
                    piece_list[idx] = internal_pieces[internal_idx]
                    internal_idx += 1
                piece_list[idx] = random.choice(self.eternity_puzzle.generate_rotation(piece_list[idx]))

        return piece_list

    def get_initial_solution(self) -> list[tuple[int, int, int, int]]:
        """
        Returns an initial solution with the same layout as the input
        """
        return copy.deepcopy(self.eternity_puzzle.piece_list)

    

def solve_local_search(eternity_puzzle: EternityPuzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    #TODO Perform random restarts
    TIME_SEARCH_SEC = 1

    time_before = time.time()
    solver = LocalSolver(eternity_puzzle)
    solver.solve()
    time_solve = time.time() - time_before
    nb_iterations = int(TIME_SEARCH_SEC / time_solve)

    best_conflict_count = solver.n_conflicts
    best_solution = solver.solution

    #TODO
    #nb_iterations = 0
    for i in range(nb_iterations):
        solver = LocalSolver(eternity_puzzle)
        solver.solve()
        if solver.n_conflicts < best_conflict_count:
            best_conflict_count = solver.n_conflicts
            best_solution = solver.solution
        print("Iteration ", i, " conflicts: ", solver.n_conflicts)

    return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)

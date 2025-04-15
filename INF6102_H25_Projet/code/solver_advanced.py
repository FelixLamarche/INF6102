import copy
import math
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

    def solve_LKH_multi_neighborhood(self, time_to_search_sec: float = 60):
        max_iterations_no_improvement = 5

        prev_conflicts = 999
        time_limit = time.time() + time_to_search_sec
        nb_iterations_no_improvement = 0
        while self.n_conflicts != prev_conflicts and time.time() < time_limit and nb_iterations_no_improvement < max_iterations_no_improvement:
            prev_conflicts = self.n_conflicts
            self.solve_LKH(time_to_search_sec)
            self.solve_LKH_complete_tiles(time_to_search_sec)
            self.solve_LKH(time_to_search_sec)
            if self.n_conflicts >= prev_conflicts:
                nb_iterations_no_improvement += 1

    def solve_LKH(self, time_to_search_sec: float = 60):
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
                if get_conflict_count_for_piece(self.solution, x1, y1, piece1, self.board_size) == 0:
                    continue
                self.do_lkh_swap(x1, y1, piece1, self.get_best_swap_LKH)
    
    def solve_LKH_complete_tiles(self, time_to_search_sec: float = 60):
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
            
            queue = board_positions

            for x1, y1 in queue:
                if time.time() > time_limit:
                    break
                piece1 = get_piece(self.solution, x1, y1, self.board_size)
                if get_conflict_count_for_piece(self.solution, x1, y1, piece1, self.board_size) == 0:
                    continue
                swapped_positions = self.do_lkh_swap(x1, y1, piece1, self.get_best_swap_LKH_complete_tiles)
                queue.extend(swapped_positions)

    def solve_best_swaps(self, time_to_search_sec: float = 60):
        """
        Solves the problem using a local search algorithm, until a local minimum is reached
        """
        # Create a neighborhood from a 2-opt, where we swap two pieces (can be a piece with another rotation of itself)
        # We select the best neighbor (the one with the lowest cost) and keep going until we reach a local minimum
        # We repeat this process from different random starts and keep the best solution found
        MIN_CONFLICTS_SOLVED = 1
        time_limit = time.time() + time_to_search_sec

        prev_conflict_count = 9999999
        while prev_conflict_count - self.n_conflicts >= MIN_CONFLICTS_SOLVED and time.time() < time_limit:
            prev_conflict_count = self.n_conflicts
            board_positions = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
            random.shuffle(board_positions)
            for x1, y1 in board_positions:
                if time.time() > time_limit:
                    break
                piece1 = get_piece(self.solution, x1, y1, self.board_size)
                if get_conflict_count_for_piece(self.solution, x1, y1, piece1, self.board_size) == 0:
                    continue
                
                piece2_to_swap, piece2_to_swap_coord, piece1_rotated = self.get_best_swap(x1, y1, piece1)
                if piece2_to_swap is not None:
                   self.swap_pieces(x1, y1, piece1_rotated, piece2_to_swap_coord[0], piece2_to_swap_coord[1], piece2_to_swap)
    
    def solve_swap_SA(self, base_temp: float, cooling: float, time_to_search_sec: float = 60):
        MAX_ITERATIONS_NO_IMPROVEMENT = 1

        temp = base_temp
        time_limit = time.time() + time_to_search_sec
        prev_conflict_count = 9999999
        iterations_no_improvement = 0
        while time.time() < time_limit and iterations_no_improvement < MAX_ITERATIONS_NO_IMPROVEMENT:
            prev_conflict_count = self.n_conflicts
            
            board_positions = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
            random.shuffle(board_positions)
            for x1, y1 in board_positions:
                if time.time() > time_limit:
                    break
                piece1 = get_piece(self.solution, x1, y1, self.board_size)
                piece2_to_swap, piece2_to_swap_coord, piece1_rotated = self.get_swap_SA(x1, y1, piece1, temp)
                if piece2_to_swap is not None:
                   self.swap_pieces(x1, y1, piece1_rotated, piece2_to_swap_coord[0], piece2_to_swap_coord[1], piece2_to_swap)
                   temp *= cooling
            
            if prev_conflict_count <= self.n_conflicts:
                iterations_no_improvement += 1
            else:
                iterations_no_improvement = 0


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

        new_piece2_conflicts = get_conflict_count_for_piece(self.solution, x1, y1, new_piece2, self.board_size)
        new_piece1_conflicts = 0
        if not are_pieces_same_position:
            new_piece1_conflicts = get_conflict_count_for_piece(self.solution, x2, y2, new_piece1, self.board_size)
        # Dont count twice the same conflict
        new_n_conflicts = new_piece2_conflicts + new_piece1_conflicts
        if has_conflict_with_adjacent_piece(x2, y2, new_piece1, x1, y1, new_piece2):
            new_n_conflicts -= 1

        # Update the running conflict count
        self.n_conflicts += new_n_conflicts - prev_n_conflicts

    def do_lkh_swap(self, x: int, y: int, piece1: tuple, swap_neighborhood) -> list:
        """
        Does a LKH swap, where we continuously swap pieces until no more beneficial swaps are found
        Returns the number of swaps done
        """
        nb_swaps = 0

        x_to_swap = x
        y_to_swap = y
        piece_to_swap = piece1
        swapped_positions = [] # Contains the positions of the pieces that have been swapped, to avoid swapping them again
        swaps = [] # Contains the pieces, and their positions

        prev_conflicts = self.n_conflicts
        best_swap_conflict_count = self.n_conflicts
        best_swap_count = -1
        while piece_to_swap is not None:
            piece2_to_swap, piece2_coord, piece_to_swap_rotated = swap_neighborhood(x_to_swap, y_to_swap, piece_to_swap, swapped_positions)
            if piece2_to_swap is None:
                break

            swapped_positions.append(piece2_coord)
            prev_piece2_rotation = get_piece(self.solution, piece2_coord[0], piece2_coord[1], self.board_size)
            swap = ((x_to_swap, y_to_swap), piece_to_swap, piece2_coord, prev_piece2_rotation)
            swaps.append(swap)
            self.swap_pieces(x_to_swap, y_to_swap, piece_to_swap_rotated, piece2_coord[0], piece2_coord[1], piece2_to_swap)

            if self.n_conflicts <= best_swap_conflict_count:
                best_swap_conflict_count = self.n_conflicts
                best_swap_count = nb_swaps

            piece_to_swap = piece2_to_swap
            nb_swaps += 1

        # Undo swaps until the best k-opt swap is reached
        for i in range(nb_swaps - 1, best_swap_count, -1):
            swap = swaps[i]
            self.swap_pieces(swap[0][0], swap[0][1], swap[3], swap[2][0], swap[2][1], swap[1])

        return swapped_positions[0: best_swap_count]
                     
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
    
    def get_best_swap_LKH(self, x: int, y: int, piece1: tuple, tabu_list_pos: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple]:
        """
        Returns the piece to swap with and its coordinates, and the correct piece rotation, if a beneficial swap is found for piece1
        Returns the best swap which reduces the most conflicts, for piece1 only and not the best swap for the whole board
        This can be used for LKH swaps to continuously swap pieces until no more beneficial swaps are found
        """
        class SwapInfo:
            # This class is used to keep track of the best swap found so far
            def __init__(self, piece1_rotation: tuple, piece2_coord: tuple[int, int], piece2_to_swap: tuple, piece1_conflict_delta: int, piece1_conflict_count: int, piece2_conflict_delta: int, piece2_conflict_count: int):
                self.piece1_rotation = piece1_rotation
                self.piece2_coord = piece2_coord
                self.piece2_to_swap = piece2_to_swap
                self.piece1_conflict_delta = piece1_conflict_delta
                self.piece1_conflict_count = piece1_conflict_count
                self.piece2_conflict_delta = piece2_conflict_delta
                self.piece2_conflict_count = piece2_conflict_count

        prev_piece1_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece1, self.board_size)
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)
        # Keep track of the best swap so far
        best_swaps = [SwapInfo(None, (-1, -1), None, 0, 4, 0, 4)]
        # Check every piece and rotation as a possible swap
        positions = [(x2, y2) for x2 in range(self.board_size) for y2 in range(self.board_size)]
        random.shuffle(positions)

        for position in positions:
            x2, y2 = position
            if (x2, y2) in tabu_list_pos:
                continue

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

                        # Check if the swap is the beneficial
                        if piece1_rotation_conflict_count < best_piece1_rotation_conflict_count or\
                            (piece1_rotation_conflict_count == best_piece1_rotation_conflict_count and piece2_rotation_conflict_count < best_piece2_conflict_count):
                            best_piece1_rotation_conflict_count = piece1_rotation_conflict_count
                            best_piece1_rotation_attempt = piece1_rotation
                            best_piece2_conflict_count = piece2_rotation_conflict_count
                            best_piece2_rotation = piece2_rotation
                # Reset the pieces to their original state
                self.solution[piece1_idx] = piece1
                self.solution[piece2_idx] = piece2
            else: # We can evaluate the swap without swapping if they are not adjacent and do not have conflicts between each other, and check less possibilities (16 to 8)
                # Get the best rotation for the original piece in piece2's place

                required_colors = get_required_colors(self.solution, x2, y2)
                if get_number_of_common_colors(piece1, required_colors) <= 4 - best_swaps[0].piece1_conflict_count:
                    continue

                temp_best_piece1_rotation_conflict_count = 999
                temp_best_piece1_rotation_attempt = None
                for piece1_rotation in piece1_rotations:
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation, self.board_size)
                    if swap_conflict_count < temp_best_piece1_rotation_conflict_count:
                        temp_best_piece1_rotation_conflict_count = swap_conflict_count
                        temp_best_piece1_rotation_attempt = piece1_rotation

                temp_best_piece2_conflict_count = 999
                temp_best_piece2_rotation = None
                # Get the best rotation for the piece2 in the original piece's place
                for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece2_rotation, self.board_size)
                    # Dont count twice the same conflict
                    if has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
                        swap_conflict_count -= 1
                    if swap_conflict_count < temp_best_piece2_conflict_count:
                        temp_best_piece2_conflict_count = swap_conflict_count
                        temp_best_piece2_rotation = piece2_rotation
                
                # Check if the swap is the beneficial
                if temp_best_piece1_rotation_conflict_count < best_piece1_rotation_conflict_count or\
                    (temp_best_piece1_rotation_conflict_count == best_piece1_rotation_conflict_count and temp_best_piece2_conflict_count < best_piece2_conflict_count):
                    best_piece1_rotation_conflict_count = temp_best_piece1_rotation_conflict_count
                    best_piece1_rotation_attempt = temp_best_piece1_rotation_attempt
                    best_piece2_conflict_count = temp_best_piece2_conflict_count
                    best_piece2_rotation = temp_best_piece2_rotation

            # Check if the swap is the beneficial and the best
            piece1_delta_conflict = best_piece1_rotation_conflict_count - prev_piece1_conflict_count
            piece2_delta_conflict = best_piece2_conflict_count - prev_piece2_conflict_count

            if piece1_delta_conflict < best_swaps[0].piece1_conflict_delta or\
                (piece1_delta_conflict == best_swaps[0].piece1_conflict_delta and piece2_delta_conflict < best_swaps[0].piece2_conflict_delta):
                # If the swap is better we replace the best swaps found so far
                best_swaps = [SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, piece1_delta_conflict, best_piece1_rotation_conflict_count, piece2_delta_conflict, best_piece2_conflict_count)]
            elif piece1_delta_conflict == best_swaps[0].piece1_conflict_delta and piece2_delta_conflict == best_swaps[0].piece2_conflict_delta:
                # If the swap is equal to the best, we add it to the list of best swaps
                best_swaps.append(SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, piece1_delta_conflict, best_piece1_rotation_conflict_count, piece2_delta_conflict, best_piece2_conflict_count))

            if best_swaps[0].piece1_conflict_count == 0 and best_swaps[0].piece2_conflict_count == 0:
                # If the swap is perfect, we can stop searching
                break
        
        chosen_swap = random.choice(best_swaps) # Randomly select one of the best swaps found
        return (chosen_swap.piece2_to_swap, chosen_swap.piece2_coord, chosen_swap.piece1_rotation)
        
    def get_best_swap_LKH_complete_tiles(self, x: int, y: int, piece1: tuple, tabu_list_pos: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple]:
        """
        Returns the piece to swap with and its coordinates, and the correct piece rotation, if a beneficial swap is found for piece1 by trying for complete tiles
        Returns the best swap which reduces the most conflicts, for piece1 only and not the best swap for the whole board
        This can be used for LKH swaps to continuously swap pieces until no more beneficial swaps are found
        """
        class SwapInfo:
            # This class is used to keep track of the best swap found so far
            def __init__(self, piece1_rotation: tuple, piece2_coord: tuple[int, int], piece2_to_swap: tuple, piece1_conflict_delta: int, piece1_conflict_count: int, piece2_conflict_delta: int, piece2_conflict_count: int):
                self.piece1_rotation = piece1_rotation
                self.piece2_coord = piece2_coord
                self.piece2_to_swap = piece2_to_swap
                self.piece1_conflict_delta = piece1_conflict_delta
                self.piece1_conflict_count = piece1_conflict_count
                self.piece2_conflict_delta = piece2_conflict_delta
                self.piece2_conflict_count = piece2_conflict_count

        prev_piece1_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece1, self.board_size)
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)
        # Keep track of the best swap so far
        best_swaps = [SwapInfo(None, (-1, -1), None, 0, 4, 0, 4)]
        # Check every piece and rotation as a possible swap
        positions = [(x2, y2) for x2 in range(self.board_size) for y2 in range(self.board_size)]
        random.shuffle(positions)

        for position in positions:
            x2, y2 = position
            if (x2, y2) in tabu_list_pos:
                continue

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

                        # Check if the swap is the beneficial
                        if piece1_rotation_conflict_count < best_piece1_rotation_conflict_count or\
                            (piece1_rotation_conflict_count == best_piece1_rotation_conflict_count and piece2_rotation_conflict_count < best_piece2_conflict_count):
                            best_piece1_rotation_conflict_count = piece1_rotation_conflict_count
                            best_piece1_rotation_attempt = piece1_rotation
                            best_piece2_conflict_count = piece2_rotation_conflict_count
                            best_piece2_rotation = piece2_rotation
                # Reset the pieces to their original state
                self.solution[piece1_idx] = piece1
                self.solution[piece2_idx] = piece2
            else: # We can evaluate the swap without swapping if they are not adjacent and do not have conflicts between each other, and check less possibilities (16 to 8)
                # Get the best rotation for the original piece in piece2's place

                temp_best_piece1_rotation_conflict_count = 999
                temp_best_piece1_rotation_attempt = None
                for piece1_rotation in piece1_rotations:
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x2, y2, piece1_rotation, self.board_size)
                    if swap_conflict_count < temp_best_piece1_rotation_conflict_count:
                        temp_best_piece1_rotation_conflict_count = swap_conflict_count
                        temp_best_piece1_rotation_attempt = piece1_rotation

                temp_best_piece2_conflict_count = 999
                temp_best_piece2_rotation = None
                # Get the best rotation for the piece2 in the original piece's place
                for piece2_rotation in self.eternity_puzzle.generate_rotation(piece2):
                    swap_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece2_rotation, self.board_size)
                    # Dont count twice the same conflict
                    if has_conflict_with_adjacent_piece(x2, y2, piece1_rotation, x, y, piece2_rotation):
                        swap_conflict_count -= 1
                    if swap_conflict_count < temp_best_piece2_conflict_count:
                        temp_best_piece2_conflict_count = swap_conflict_count
                        temp_best_piece2_rotation = piece2_rotation
                
                # Check if the swap is the beneficial
                if temp_best_piece1_rotation_conflict_count < best_piece1_rotation_conflict_count or\
                    (temp_best_piece1_rotation_conflict_count == best_piece1_rotation_conflict_count and temp_best_piece2_conflict_count < best_piece2_conflict_count):
                    best_piece1_rotation_conflict_count = temp_best_piece1_rotation_conflict_count
                    best_piece1_rotation_attempt = temp_best_piece1_rotation_attempt
                    best_piece2_conflict_count = temp_best_piece2_conflict_count
                    best_piece2_rotation = temp_best_piece2_rotation

            # Check if the swap is the beneficial and the best
            piece1_delta_conflict = best_piece1_rotation_conflict_count - prev_piece1_conflict_count
            piece2_delta_conflict = best_piece2_conflict_count - prev_piece2_conflict_count

            if best_piece1_rotation_conflict_count == 0 and best_piece2_conflict_count == 0:
                # If the swap is perfect, we can stop searching
                best_swaps = [SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, piece1_delta_conflict, best_piece1_rotation_conflict_count, piece2_delta_conflict, best_piece2_conflict_count)]
                break

            if best_piece1_rotation_conflict_count == 0:
                best_swaps.append(SwapInfo(best_piece1_rotation_attempt, (x2, y2), best_piece2_rotation, piece1_delta_conflict, best_piece1_rotation_conflict_count, piece2_delta_conflict, best_piece2_conflict_count))

        if len(best_swaps) == 0:
            return (None, (-1, -1), None)
        chosen_swap = random.choice(best_swaps) # Randomly select one of the best swaps found
        return (chosen_swap.piece2_to_swap, chosen_swap.piece2_coord, chosen_swap.piece1_rotation)

    def get_swap_SA(self, x: int, y: int, piece1: tuple, temp: float) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple]:
        """
        Returns the piece to swap with and its coordinates, and the correct piece rotation, if a beneficial swap is found
        """
        prev_piece1_conflict_count = get_conflict_count_for_piece(self.solution, x, y, piece1, self.board_size)
        piece1_rotations = self.eternity_puzzle.generate_rotation(piece1)

        # Check every piece and rotation as a possible swap randomly so we can pick the first one without bias
        positions = [(x2, y2) for x2 in range(self.board_size) for y2 in range(self.board_size)]
        random.shuffle(positions)

        for x2, y2 in positions:
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
            old_conflict_count = prev_piece1_conflict_count + prev_piece2_conflict_count
            new_conflict_count = best_piece2_conflict_count + best_piece1_rotation_conflict_count
            delta_conflict_count = new_conflict_count - old_conflict_count
            if delta_conflict_count < 0 or self.probability_annealing(temp, old_conflict_count, new_conflict_count):
                return (best_piece2_rotation, (x2, y2), best_piece1_rotation_attempt)
        
        return (None, (-1, -1), None)

    def probability_annealing(self, temp: float, old_conflicts: int, new_conflicts: int) -> bool:
        """Returns True if we should accept according to the simulated annealing probability"""
        if old_conflicts > new_conflicts:
            return True
        f = math.exp(-(new_conflicts - old_conflicts) / temp)
        return random.random() < f
    
    def set_tiles_conflict_counts(self):
        """
        Sets the conflict counts for each tile in the solution
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                idx = get_idx(x, y, self.board_size)
                self.tile_conflict_counts[idx] = get_conflict_count_for_piece(self.solution, x, y, self.solution[idx], self.board_size)


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

def get_initial_solution_heuristic_best(eternity_puzzle: EternityPuzzle, time_to_search: float) -> list[tuple[int, int, int, int]]:
    """
    Returns the best solution found by the heuristic solver
    """
    best_solution = None
    best_conflict_count = 999
    end_time = time.time() + time_to_search

    while time.time() < end_time:
        solution = solve_heuristic(eternity_puzzle)[0]
        conflict_count = eternity_puzzle.get_total_n_conflict(solution)
        if conflict_count < best_conflict_count:
            best_conflict_count = conflict_count
            best_solution = solution
        if conflict_count == 0:
            break
    return best_solution

def solve_backtrack(eternity_puzzle: EternityPuzzle, search_time: float) -> list[tuple[int, int, int, int]]:
    """
    Returns a solution using backtracking
    """
    piece_list = copy.deepcopy(eternity_puzzle.piece_list)
    solution = [None] * (eternity_puzzle.board_size ** 2)
    solution = solve_backtrack_recursive(eternity_puzzle, solution, piece_list, 0, 0, search_time + time.time())
    return solution

def solve_backtrack_recursive(eternity_puzzle: EternityPuzzle, solution: list[tuple[int, int, int, int]], remaining_pieces: list[tuple[int, int, int, int]], x: int, y: int, time_limit_sec: float):
    """
    Returns a solution using backtracking
    """
    idx = get_idx(x, y, eternity_puzzle.board_size)
    for piece in remaining_pieces:
        if time.time() > time_limit_sec:
            # Add the remaining pieces to the solution
            for i in range(len(remaining_pieces)):
                solution[idx + i] = remaining_pieces[i]
            return solution
        piece_rotations = eternity_puzzle.generate_rotation(piece)
        for rotation in piece_rotations:
            conflict_count = get_conflict_count_for_piece_incomplete(solution, x, y, rotation, eternity_puzzle.board_size)
            if conflict_count == 0:
                solution[idx] = rotation
                piece_list = copy.deepcopy(remaining_pieces)
                piece_list.remove(piece)

                next_x = x + 1
                next_y = y
                if next_x >= eternity_puzzle.board_size:
                    next_x = 0
                    next_y += 1
                
                # Terminal condition
                if next_y >= eternity_puzzle.board_size:
                    return solution
                #print("Backtrack: ", x, y, " piece: ", piece, " rotation: ", rotation)
                result = solve_backtrack_recursive(eternity_puzzle, solution, piece_list, next_x, next_y, time_limit_sec)
                if result is not None:
                    return result
                
                solution[idx] = None
    return None

def shuffle_solution_random(solution: list[tuple[int, int, int, int]], nb_pieces_to_shuffle: int) -> list[tuple[int, int, int, int]]:
    """
    Copies and shuffles the solution by randomly swapping a specific amount of pieces between each other
    In essence a perturbation method for an iterative local search
    """
    board_size = int(len(solution) ** 0.5)
    positions = [(x, y) for x in range(board_size) for y in range(board_size)]
    positions_to_shuffle = random.sample(positions, k=nb_pieces_to_shuffle)
    new_solution = copy.deepcopy(solution)
    
    # Shuffle by doing a k-opt swap
    for i in range(nb_pieces_to_shuffle):
        x1, y1 = positions_to_shuffle[i]
        x2, y2 = positions_to_shuffle[(i + 1) % nb_pieces_to_shuffle]
        idx1 = get_idx(x1, y1, board_size)
        idx2 = get_idx(x2, y2, board_size)
        temp = new_solution[idx1]
        new_solution[idx1] = new_solution[idx2]
        new_solution[idx2] = temp

    # Then do random 2-opt swaps
    while len(positions_to_shuffle) > 1:
        random_pos1 = random.choice(positions_to_shuffle)
        positions_to_shuffle.remove(random_pos1)
        random_pos2 = random.choice(positions_to_shuffle)
        positions_to_shuffle.remove(random_pos2)
        
        x1, y1 = random_pos1
        x2, y2 = random_pos2
        idx1 = get_idx(x1, y1, board_size)
        idx2 = get_idx(x2, y2, board_size)
        temp = new_solution[idx1]
        new_solution[idx1] = new_solution[idx2]
        new_solution[idx2] = temp

    
    return new_solution

def shuffle_solution(solution: list[tuple[int, int, int, int]], nb_shuffle_no_conflicts: int, nb_shuffle_conflicts: int) -> list[tuple[int, int, int, int]]:
    """
    Copies and shuffles the solution by randomly swapping a specific amount of pieces with conflicts and without conflicts between each other
    In essence a perturbation method for an iterative local search
    """
    board_size = int(len(solution) ** 0.5)

    positions_conflicts = []
    positions_no_conflicts = []
    for x in range(board_size):
        for y in range(board_size):
            idx = get_idx(x, y, board_size)
            piece = solution[idx]
            if get_conflict_count_for_piece(solution, x, y, piece, board_size) > 0:
                positions_conflicts.append((x, y))
            else:
                positions_no_conflicts.append((x, y))
    if nb_shuffle_conflicts > len(positions_conflicts):
        nb_shuffle_no_conflicts += nb_shuffle_conflicts - len(positions_conflicts)
        nb_shuffle_conflicts = len(positions_conflicts)
    if nb_shuffle_no_conflicts > len(positions_no_conflicts):
        nb_shuffle_conflicts += nb_shuffle_no_conflicts - len(positions_no_conflicts)
        nb_shuffle_no_conflicts = len(positions_no_conflicts)

    positions_to_shuffle = random.sample(positions_conflicts, k=nb_shuffle_conflicts) + random.sample(positions_no_conflicts, k=nb_shuffle_no_conflicts)
    random.shuffle(positions_to_shuffle)
    new_solution = copy.deepcopy(solution)
    
    # Do a k-opt swap
    for i in range(len(positions_to_shuffle)):
        x1, y1 = positions_to_shuffle[i]
        x2, y2 = positions_to_shuffle[(i + 1) % len(positions_to_shuffle)]
        idx1 = get_idx(x1, y1, board_size)
        idx2 = get_idx(x2, y2, board_size)
        temp = new_solution[idx1]
        new_solution[idx1] = new_solution[idx2]
        new_solution[idx2] = temp

    # Then do random 2-opt swaps
    while False and len(positions_to_shuffle) > 1:
        random_pos1 = random.choice(positions_to_shuffle)
        positions_to_shuffle.remove(random_pos1)
        random_pos2 = random.choice(positions_to_shuffle)
        positions_to_shuffle.remove(random_pos2)

        x1, y1 = random_pos1
        x2, y2 = random_pos2
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
    TIME_SEARCH_MARGIN = 15 # PUT This to 5 seconds to avoid timing out in the final version
    TIME_SEARCH_HEURISTIC = TIME_SEARCH_SEC * 0.02 # 2% of the time for the heuristic

    NB_ITERATIONS_NO_IMPROVEMENT = 250

    best_solution = None # Best solution found so far
    best_conflict_count = 999
    best_solution_cur_search = None # Best solution found in the current search
    best_conflict_count_cur_search = 999

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
            # Chance to either restart a new search, or resume from the best solution found so far
            print("Restarting search: Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts_cur_search: ", best_conflict_count_cur_search)
            if random.uniform(0.0, 1.0) < 0.5:
                best_solution_cur_search = None 
                best_conflict_count_cur_search = 999
            else:
                best_solution_cur_search = best_solution
                best_conflict_count_cur_search = best_conflict_count
            iteration_without_improvement = 0
                

        # Get an initial solution
        initial_solution = best_solution_cur_search
        nb_pieces_to_shuffle = 0
        ratio_of_conflicts = 0
        if initial_solution is None: # If there isn't an initial solution
            initial_solution = get_initial_solution_heuristic_best(eternity_puzzle, min(TIME_SEARCH_HEURISTIC, time_to_search))
            #initial_solution = solve_backtrack(eternity_puzzle, 5.0)
            #initial_solution = get_initial_solution_semi_random(eternity_puzzle)
            print("Initial solution conflicts: ", eternity_puzzle.get_total_n_conflict(initial_solution))
        else: # If there is an initial solution, perturb it
            percentage_of_shuffle = random.uniform(0.05, 0.50)
            nb_pieces_to_shuffle = int(len(initial_solution) * percentage_of_shuffle)
            ratio_of_conflicts = random.uniform(0.2, 0.5)
            nb_shuffle_pieces_conflicts = int(nb_pieces_to_shuffle * ratio_of_conflicts)
            nb_shuffle_pieces_no_conflicts = nb_pieces_to_shuffle - nb_shuffle_pieces_conflicts
            initial_solution = shuffle_solution(initial_solution, nb_shuffle_pieces_no_conflicts, nb_shuffle_pieces_conflicts)

        solver = AdvancedSolver(eternity_puzzle, initial_solution)
        random_choice = random.randint(0, 0)
        solver.solve_LKH(time_to_search)

        # if random_choice == 0:
            # solver.solve_LKH(time_to_search)
        #     #solver.solve_LKH_complete_tiles(time_to_search)
        #     #solver.solve_LKH_multi_neighborhood(time_to_search)
        # elif random_choice == 1:
        #     solver.solve_best_swaps(time_to_search)
        # else:
        #     temp = 1
        #     cooling_factor = 0.90
        #     solver.solve_swap_SA(temp, cooling_factor, time_to_search)


        iteration_count += 1
        iteration_without_improvement += 1

        if solver.n_conflicts < best_conflict_count_cur_search:
            best_conflict_count_cur_search = solver.n_conflicts
            best_solution_cur_search = solver.solution
            iteration_without_improvement = 0
            #print("NewLocalBest : Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter), " nb pieces shuffled: ", nb_pieces_to_shuffle)

        #print("Conflicts: ", solver.n_conflicts)
        if solver.n_conflicts < best_conflict_count:
            best_conflict_count = solver.n_conflicts
            best_solution = solver.solution
            print("NewGlobalBest: Iteration:", iteration_count, " time: ", "{:.2f}".format(time.time() - time_before), " best_conflicts: ", best_conflict_count, \
                  " conflicts: ", solver.n_conflicts, " time iter: ", "{:.2f}".format(time.time() - time_start_iter), " nb pieces shuffled: ", nb_pieces_to_shuffle, \
                    " ratio of conflicts: ", "{:.2f}".format(ratio_of_conflicts), " random choice: ", random_choice)


        if best_conflict_count == 0:
            break

    return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)

import copy
import numpy as np

def solve_heuristic(eternity_puzzle) -> tuple[list[tuple[int, int, int, int]], int]:
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    """
    DESCRIPTION:
    1. Start from a corner and place each successive tile in a spiral pattern.
    So each tile is placed next to the previous one.
    The early tiles could be placed without much errors since there are less constraints.
    So for each next tile to choose, we can verify each tile in each orientation and choose the one that generates the least conflicts.

    piece order: North, South, West, East
    """

    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3

    GRAY = 0

    GRAY_PIECE = (GRAY, GRAY, GRAY, GRAY)


    def get_idx(x, y):
        return y * eternity_puzzle.board_size + x
    
    # Gets the next coordinate to place the piece with a spiral pattern starting from the lower-left corner
    def range_spiral(square_size):
        x = 0
        y = 0
        dx = 1
        dy = 0
        x_end_pos = square_size - 1
        x_end_neg = 0
        y_end_pos = square_size - 1
        y_end_neg = 1
        for _ in range(square_size * square_size):

            yield x, y
            
            x += dx
            y += dy

            if dx > 0 and x == x_end_pos:
                dx = 0
                dy = 1
                x_end_pos -= 1
            elif dx < 0 and x == x_end_neg:
                dx = 0
                dy = -1
                x_end_neg += 1
            elif dy > 0 and y == y_end_pos:
                dx = -1
                dy = 0
                y_end_pos -= 1
            elif dy < 0 and y == y_end_neg:
                dx = 1
                dy = 0
                y_end_neg += 1


    def is_corner_piece(piece):
        return piece.count(GRAY) == 2

    def is_edge_piece(piece):
        return piece.count(GRAY) == 1

    def is_corner(x, y):
        return (x == 0 or x == eternity_puzzle.board_size - 1) and (y == 0 or y == eternity_puzzle.board_size - 1)

    def is_edge(x, y):
        return x == 0 or x == eternity_puzzle.board_size - 1 or y == 0 or y == eternity_puzzle.board_size - 1

    def get_piece(solution, x, y):
        if x < 0 or x >= eternity_puzzle.board_size or y < 0 or y >= eternity_puzzle.board_size:
            return GRAY_PIECE
        return solution[get_idx(x, y)]

    def get_conflict_count(solution, x, y, piece):
        n_conflict = 0

        if is_corner(x, y):
            if not is_corner_piece(piece):
                n_conflict += 10 # non-corner pieces should not be in the corner
        elif is_edge(x, y) and not is_edge_piece(piece):
                n_conflict += 10 #non-edge pieces should not be in the edge

        # Check NORTH
        if piece[NORTH] != get_piece(solution, x, y + 1)[SOUTH]:
            n_conflict += 1
        if piece[SOUTH] != get_piece(solution, x, y - 1)[NORTH]:
            n_conflict += 1
        if piece[WEST] != get_piece(solution, x - 1, y)[EAST]:
            n_conflict += 1
        if piece[EAST] != get_piece(solution, x + 1, y)[WEST]:
            n_conflict += 1
        return n_conflict
    
    # starts from the lower-left corner
    solution = [(-1, -1, -1, -1) for _ in range(eternity_puzzle.n_piece)]

    remaining_pieces = copy.deepcopy(eternity_puzzle.piece_list)

    # We start from the bottom-left, go to the bottom-right, then top-right, and finally top-left, and keep going in this spiral pattern
    for x,y in range_spiral(eternity_puzzle.board_size):
        best_piece_to_remove = remaining_pieces[0]
        best_piece_rotated = remaining_pieces[0]
        least_conflict_count = 999999
        for piece in remaining_pieces:
            piece_permutations = eternity_puzzle.generate_rotation(piece)
            for permutation in piece_permutations:
                n_conflict = get_conflict_count(solution, x, y, permutation)
                if n_conflict < least_conflict_count:
                    least_conflict_count = n_conflict
                    best_piece_rotated = permutation
                    best_piece_to_remove = piece
        
        solution[get_idx(x, y)] = best_piece_rotated
        remaining_pieces.remove(best_piece_to_remove)

    return solution, eternity_puzzle.get_total_n_conflict(solution)
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

GRAY = 0

GRAY_PIECE = (GRAY, GRAY, GRAY, GRAY)

def get_idx(x: int, y: int, board_size: int) -> int:
    return y * board_size + x

def is_corner_piece(piece: tuple[int, int, int, int]) -> bool:
    return piece.count(GRAY) == 2

def is_edge_piece(piece: tuple[int, int, int, int]) -> bool:
    return piece.count(GRAY) == 1

def is_corner(x: int, y: int, board_size: int) -> bool:
    return (x == 0 or x == board_size - 1) and (y == 0 or y == board_size - 1)

def is_edge(x: int, y: int, board_size: int) -> bool:
    return x == 0 or x == board_size - 1 or y == 0 or y == board_size - 1

def get_piece(solution: list[tuple[int, int, int, int]], x: int, y: int, board_size: int) -> tuple[int, int, int, int]:
    if x < 0 or x >= board_size or y < 0 or y >= board_size:
        return GRAY_PIECE
    return solution[get_idx(x, y, board_size)]

def are_coords_adjacent(x1: int, y1: int, x2: int, y2: int) -> bool:
    return abs(x1 - x2) + abs(y1 - y2) == 1

def has_conflict_with_adjacent_piece(x1:int, y1:int, piece1:tuple, x2:int, y2:int, piece2:tuple) -> bool:
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

def get_conflict_count_for_piece(solution: list[tuple[int, int, int, int]], x: int, y: int, piece: tuple, board_size: int) -> int:
    """
    Returns the number of conflicts that a piece would have if placed at the position
    """
    n_conflict = 0
    # Check NORTH
    if piece[NORTH] != get_piece(solution, x, y + 1, board_size)[SOUTH]:
        n_conflict += 1
    if piece[SOUTH] != get_piece(solution, x, y - 1, board_size)[NORTH]:
        n_conflict += 1
    if piece[WEST] != get_piece(solution, x - 1, y, board_size)[EAST]:
        n_conflict += 1
    if piece[EAST] != get_piece(solution, x + 1, y, board_size)[WEST]:
        n_conflict += 1
    return n_conflict
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
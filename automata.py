import numpy as np

def place_glider(state_map, x, y):
    """
    Place a glider pattern on the state_map with top-left corner at (x, y).
    
    Args:
        state_map (np.ndarray): The 2D grid of the automaton.
        x (int): x-coordinate (column) of top-left corner.
        y (int): y-coordinate (row) of top-left corner.
    """
    glider_pattern = np.array([
        [0,1,0],
        [0,0,1],
        [1,1,1]
    ], dtype=np.uint8)

    h, w = state_map.shape
    ph, pw = glider_pattern.shape

    # Check bounds to avoid IndexError
    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        raise ValueError("Glider placement out of bounds")

    # Place the pattern on the state_map
    state_map[y:y+ph, x:x+pw] = glider_pattern

#place_glider(state_map, 10, 10)  # puts a glider at column=10, row=10

def place_block(state_map, x, y):
    block = np.array([
        [1,1],
        [1,1]
    ], dtype=np.uint8)

    h, w = state_map.shape
    ph, pw = block.shape

    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        raise ValueError("Block placement out of bounds")

    state_map[y:y+ph, x:x+pw] = block

def place_blinker(state_map, x, y):
    blinker = np.array([
        [1,1,1]
    ], dtype=np.uint8)

    h, w = state_map.shape
    ph, pw = blinker.shape

    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        raise ValueError("Blinker placement out of bounds")

    state_map[y:y+ph, x:x+pw] = blinker

def place_pattern(state_map, x, y, pattern):
    h, w = state_map.shape
    ph, pw = pattern.shape

    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        raise ValueError("Pattern placement out of bounds")

    state_map[y:y+ph, x:x+pw] = pattern

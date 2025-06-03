import numpy as np
from display import display_toggle_loop

# === Main ===

def main():
    width, height = 200, 200
    scale_factor = 4
    state_map = np.zeros((height, width), dtype=np.uint8)
    final_map = display_toggle_loop(state_map, scale_factor)

if __name__ == "__main__":
    main()
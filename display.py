import cv2
import numpy as np
import time
from automata import place_glider, place_block, place_blinker, place_pattern

# === Utility Functions ===

def toggle_pixel(state_map, x, y):
    state_map[y, x] = 1 - state_map[y, x]  # Toggle between 0 and 1

def create_image_state(width, height, state_map):
    return (state_map * 255).astype(np.uint8)

def resize_for_display(img, scale_factor):
    return cv2.resize(
        img,
        (img.shape[1] * scale_factor, img.shape[0] * scale_factor),
        interpolation=cv2.INTER_NEAREST
    )

def apply_temperature_step(state_map, targets):
    for x, y in targets:
        if 0 <= x < state_map.shape[1] and 0 <= y < state_map.shape[0]:
            state_map[y, x] = 1  # Simplified placeholder logic

def game_of_life_step(state_map):
    new_map = state_map.copy()
    for y in range(1, state_map.shape[0] - 1):
        for x in range(1, state_map.shape[1] - 1):
            live_neighbors = np.sum(state_map[y-1:y+2, x-1:x+2]) - state_map[y, x]
            
            # Apply Conway's rules
            if state_map[y, x] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    new_map[y, x] = 0
                # Additional survival rule: survives if exactly 5 neighbors
                #elif live_neighbors == 5:
                    #new_map[y, x] = 1
            else:
                # OG birth rule
                if live_neighbors == 3:
                    new_map[y, x] = 1
                # Additional birth rule: born if exactly 6 neighbors
                #elif live_neighbors == 6:
                    #new_map[y, x] = 1
                # Death rule
                elif live_neighbors == 7:
                    new_map[y, x] = 0
    return new_map

# === Display Logic ===

def setup_display_window(window_name, state_map, scale_factor, redraw_flag):
    def mouse_callback(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            grid_x = x // scale_factor
            grid_y = y // scale_factor
            print(f"Click at ({x},{y}) => grid ({grid_x},{grid_y})")
            if 0 <= grid_x < state_map.shape[1] and 0 <= grid_y < state_map.shape[0]:
                toggle_pixel(state_map, grid_x, grid_y)
                apply_temperature_step(state_map, [(grid_x, grid_y)])
                redraw_flag[0] = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

def run_game_loop(state_map, window_name, scale_factor, redraw_flag):
    
    h, w = state_map.shape
    while True:
        if redraw_flag[0]:
            img = create_image_state(state_map.shape[1], state_map.shape[0], state_map)
            img_large = resize_for_display(img, scale_factor)
            cv2.imshow(window_name, img_large)
            redraw_flag[0] = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state_map[np.random.randint(0, state_map.shape[0], 200),
                      np.random.randint(0, state_map.shape[1], 200)] = 1
            redraw_flag[0] = True
        elif key == ord('n'):
            state_map[:] = game_of_life_step(state_map)
            redraw_flag[0] = True
        elif key == ord('g'):
            place_glider(state_map, h//2, w//2) # puts a glider at center
            redraw_flag[0] = True
        else:
            time.sleep(0.3)
            state_map[:] = game_of_life_step(state_map)
            redraw_flag[0] = True

    cv2.destroyAllWindows()
    return state_map

def display_toggle_loop(state_map, scale_factor):
    window_name = "Toggle Pixels"
    redraw_flag = [True]  # Mutable flag for triggering redraws

    setup_display_window(window_name, state_map, scale_factor, redraw_flag)
    return run_game_loop(state_map, window_name, scale_factor, redraw_flag)


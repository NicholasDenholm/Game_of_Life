import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel
from PIL import Image
import cv2
import random
import time

# ~~~~~~~~~~~~~~~~~ Image creation ~~~~~~~~~~~~~~~~~ #

def create_pillow_image(width, height, color=(0, 0, 0)):
    """Create a blank RGB image of given size and background color."""
    array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(array)

def create_image(width, height, color=(0, 0, 0)):
    array = np.full((height, width, 3), color, dtype=np.uint8)
    return array  # NumPy array, not a PIL image

def create_image_state(width, height, state_map):
    """Create an image from a binary state map (0=black, 1=white)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[state_map == 1] = [255, 255, 255]  # White for 'on' pixels
    return img

# ________________ Changing pixels _________________ #

def toggle_pixel(state_map, x, y):
    """Toggle pixel state in the state map (0 -> 1, 1 -> 0)."""
    if 0 <= x < state_map.shape[1] and 0 <= y < state_map.shape[0]:
        state_map[y, x] = 1 - state_map[y, x]  # Flip state
    return state_map

def toggle_nearby_pixel(state_map, x, y, temperature):
    """Toggle pixel state in the state map (0 -> 1, 1 -> 0)."""

    if 0 <= x < state_map.shape[1] and 0 <= y < state_map.shape[0]:
        state_map[y, x] = 1 - state_map[y, x]  # Flip state
    return state_map

def modify_pixel(img, x, y, color):
    """Modify a single pixel in the image to a given RGB color."""
    #array = np.array(img)
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] = color
    return img

def random_pixel_change(state_map, count=10, mode='toggle'):
    height, width = state_map.shape
    new_map = state_map.copy()

    for px in range(count):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        
        
        if mode == "toggle":
            new_map = toggle_pixel(new_map, x, y)
        elif mode == "on":
            new_map[y, x] = 1
        elif mode == "off":
            new_map[y, x] = 0

        return new_map

# ----------------------- Display ----------------------- #

def show_pillow_image(img):
    Image.Image.show(img)


def display_toggle_loop(state_map):
    """Display the image and toggle pixels on mouse click."""
    window_name = "Toggle Pixels"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            toggle_pixel(state_map, x, y)

            targets = [(x,y)]
            apply_temperature_step(state_map, targets)

            targets = targets = [(x+1,y+1)]
            apply_temperature_step(state_map, targets)
            #game_of_life_step(state_map)
            updated_img = create_image_state(state_map.shape[1], state_map.shape[0], state_map)

            #zoomed_img = zoom_image(updated_img, center_x=50, center_y=50, zoom_factor=2)
            #cv2.imshow("Zoomed View", zoomed_img)

            cv2.imshow(window_name, updated_img)
            img = updated_img
            #cv2.waitKey(10)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img = create_image_state(state_map.shape[1], state_map.shape[0], state_map)
        
        #zoomed_img = zoom_image(img, center_x=50, center_y=50, zoom_factor=2)
        
        cv2.imshow(window_name,img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            state_map = random_pixel_change(state_map, count=20, mode="on")
        elif key == ord('n'):
            # Advance one step in Game of Life
            state_map = game_of_life_step(state_map)
        else:

            time.sleep(0.3)
            state_map = game_of_life_step(state_map)

    cv2.destroyAllWindows()
    return state_map


def zoom_image(img, center_x, center_y, zoom_factor=2):
    h, w = img.shape[:2]

    zoom_w = w // zoom_factor
    zoom_h = h // zoom_factor

    x1 = max(center_x - zoom_w // 2, 0)
    y1 = max(center_y - zoom_h // 2, 0)
    x2 = min(x1 + zoom_w, w)
    y2 = min(y1 + zoom_h, h)

    cropped = img[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return zoomed


def display_loop_update(img_array):
    """Display the image and register mouse clicks to update pixels."""
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at: ({x}, {y})")
            modify_pixel(img_array, x, y, (0, 0, 255))  # Red dot on click
            cv2.imshow("Clickable Image", img_array)

    window_name = "Clickable Image"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, img_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Call method to check for red dots


    cv2.destroyAllWindows()

# +++++++++++++++ Detecting +++++++++++++++++++ #

def find_red_dot(img_array):
    return None

def find_white_dot(state_map):
    return None


# ............... Math ................. # 


def fibonacci_up_to_limit(limit: int):
    fib_sequence = []
    a, b = 0, 1  # Starting with 0 and 1
    while a <= limit:
        fib_sequence.append(a)
        a, b = b, a + b  # Update the values for the next Fibonacci number
    return fib_sequence


def apply_temperature_step(state_map, targets=None):
    """
    Simple smoothing: each cell becomes the average of itself and its 8 neighbors.
    If targets is None, update all pixels.
    Rounded to the nearest integer.
    | x x x |
    | x 0 x |
    | x x x |
    """

    height, width = state_map.shape
    new_map = state_map.copy()

    # If no targets provided, update the whole grid
    if targets is None:
        targets = [(x, y) for y in range(height) for x in range(width)]

    for y in range(height):
        for x in range(width):
            # Get neighbors including self
            values = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        values.append(state_map[ny, nx])
            # Average and round
            new_map[y, x] = int(round(sum(values) / len(values)))
    
    return new_map

def game_of_life_step(state_map):
    height, width = state_map.shape
    new_map = np.zeros_like(state_map)

    for y in range(height):
        for x in range(width):
            # Count live neighbors
            live_neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if dy == 0 and dx == 0:
                        continue  # Skip self
                    if 0 <= ny < height and 0 <= nx < width:
                        live_neighbors += state_map[ny, nx]

            # Apply Conway's rules
            if state_map[y, x] == 1:
                if live_neighbors in [2, 3]:
                    new_map[y, x] = 1
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

    return new_map

# +++++++++++++++ MAIN +++++++++++++++++++ #

def main():
    print("hello!")
    option = 0

    if option == 1:
            
        width, height = 100, 100
        background_color = (0, 0, 0)  # black

        img = create_image(width, height, background_color)

        # Modify a pixel at (50, 50) to red
        #img = modify_pixel(img, 50, 50, (255, 0, 0))

        # Display image in a loop
        display_loop_update(img)
    else:
        width, height = 250, 250
        state_map = np.zeros((height, width), dtype=np.uint8)  # All pixels start off (0)
        final_map = display_toggle_loop(state_map)

        print("Final toggled pixel coordinates:")
        on_pixels = np.argwhere(final_map == 1)
        for y, x in on_pixels:
            print(f"({x}, {y})")

if __name__ == "__main__":
    main()
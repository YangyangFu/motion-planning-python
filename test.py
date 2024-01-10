import cv2
import numpy as np

# Constants
GRID_SIZE = 20
GRID_WIDTH = 40  # Change this value according to the desired window size

# Global variables for image and window name
img = np.ones((GRID_SIZE * GRID_WIDTH, GRID_SIZE * GRID_WIDTH, 3), np.uint8) * 255
window_name = 'Grid'

# Function to draw the grid
def draw_grid():
    global img
    rows, cols, _ = img.shape
    for i in range(0, rows, GRID_SIZE):
        cv2.line(img, (0, i), (cols, i), (255, 255, 255), 1)
    for j in range(0, cols, GRID_SIZE):
        cv2.line(img, (j, 0), (j, rows), (255, 255, 255), 1)

# Function to handle mouse events
def draw_color(event, x, y, flags, param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        row = y // GRID_SIZE
        col = x // GRID_SIZE

        # Toggle grid color
        if img[row * GRID_SIZE, col * GRID_SIZE, 0] == 255:
            img[row * GRID_SIZE:(row + 1) * GRID_SIZE, col * GRID_SIZE:(col + 1) * GRID_SIZE] = [0, 0, 0]
        else:
            img[row * GRID_SIZE:(row + 1) * GRID_SIZE, col * GRID_SIZE:(col + 1) * GRID_SIZE] = [255, 255, 255]

        cv2.imshow(window_name, img)

# Function to create the window and start the grid display
def start_grid():
    global img

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_color)

    # Draw initial grid
    draw_grid()
    cv2.imshow(window_name, img)

# Function to draw a line on the current OpenCV window
def draw_line(point1, point2):
    global img

    cv2.line(img, point1, point2, (0, 0, 0), 2)
    cv2.imshow(window_name, img)

# Use start_grid() to initialize the window and display the grid
start_grid()

# Example of drawing a line between two points
# You can call this function as needed with different point coordinates
draw_line((20, 20), (100, 100))

cv2.waitKey(0)
cv2.destroyAllWindows()

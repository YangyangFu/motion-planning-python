import cv2
import numpy as np
import imageio

# Constants
GRID_SIZE = 20
GRID_WIDTH = 40  # Change this value according to the desired window size
OUTPUT_GIF = 'grid_experiment.gif'

# Function to draw the grid
def draw_grid(img):
    rows, cols, _ = img.shape
    for i in range(0, rows, GRID_SIZE):
        cv2.line(img, (0, i), (cols, i), (255, 255, 255), 1)
    for j in range(0, cols, GRID_SIZE):
        cv2.line(img, (j, 0), (j, rows), (255, 255, 255), 1)

# Function to handle mouse events
def draw_color(event, x, y, flags, param):
    global img, frames

    if event == cv2.EVENT_LBUTTONDOWN:
        row = y // GRID_SIZE
        col = x // GRID_SIZE

        # Toggle grid color
        if img[row * GRID_SIZE, col * GRID_SIZE, 0] == 255:
            img[row * GRID_SIZE:(row + 1) * GRID_SIZE, col * GRID_SIZE:(col + 1) * GRID_SIZE] = [0, 0, 0]
        else:
            img[row * GRID_SIZE:(row + 1) * GRID_SIZE, col * GRID_SIZE:(col + 1) * GRID_SIZE] = [255, 255, 255]

        # Save current frame
        frames.append(np.copy(img))

        cv2.imshow('Grid', img)

# Create a blank image
img = np.ones((GRID_SIZE * GRID_WIDTH, GRID_SIZE * GRID_WIDTH, 3), np.uint8) * 255
frames = [np.copy(img)]
cv2.namedWindow('Grid')
cv2.setMouseCallback('Grid', draw_color)

# Draw initial grid
draw_grid(img)
cv2.imshow('Grid', img)

while True:
    key = cv2.waitKey(1)
    if key == 27:  # Press 'esc' to exit
        break

cv2.destroyAllWindows()

# Save frames as a GIF
imageio.mimsave(OUTPUT_GIF, frames, duration=0.1)
import cv2
import numpy as np

# Create a black image
width, height = 200, 200
img = np.zeros((height, width, 3), dtype=np.uint8)

# Function to plot a pixel as a square with a given color
def plot_pixel(image, x, y, color):
    cv2.rectangle(image, (x, y), (x + 1, y + 1), color, -1)

# Example: Plot a pixel at position (50, 50) with red color (BGR: 0, 0, 255)
print(img[50, 50])
plot_pixel(img, 50, 50, (0, 0, 255))
print(img[50,50])
# Display the image with the plotted pixel
cv2.imshow('Pixel as Square', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

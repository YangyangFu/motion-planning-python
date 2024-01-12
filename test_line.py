import cv2
import numpy as np

# Create a black image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Define the pixel coordinates for the center of the star
center_coordinates = (250, 250)

# Define the color of the star (BGR format)
color = (0, 255, 0)  # Green color

# Draw the star on the image
cv2.drawMarker(image, center_coordinates, color, markerType=cv2.MARKER_STAR, markerSize=50, thickness=1)

# Display the image
cv2.imshow('Star Example', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

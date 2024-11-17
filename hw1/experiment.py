import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread('input_part1.jpg')

# result with cv2.GaussianBlur and cv2.medianBlur
for k in [21, 41, 61, 81, 101, 121]:

    blurred_image = cv2.GaussianBlur(image, (k, k), 51)
    # blurred_image = cv2.medianBlur(image, k)
    cv2.imwrite(f"kernel{k}_cv2.jpg", blurred_image)

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from Convolution import convolution
from Gausian_Smoothing import gaussian_blur

def sobel_edge_detection(image, filter, convert_to_degree=False, verbose=False):
    new_image_x = convolution(image, filter, verbose)
    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()
    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title('Vertical Edge')
        plt.show()
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
    
    gradient_direction = np.arctan2(new_image_x, new_image_y)
    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180
        
    return gradient_magnitude, gradient_direction

if __name__ == "__main__":
    filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="//img_test//lena.png")
    # args = vars(ap.parse_args())
    
    image = cv2.imread(".//img_test//lena.png")
    image = gaussian_blur(image, 9, verbose=True)
    sobel_edge_detection(image, filter, verbose=True)
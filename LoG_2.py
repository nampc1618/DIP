import cv2
import numpy as np
from Convolution import convolution
import Mask
import matplotlib.pyplot as plt

def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0 
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
            
            # If both negative and positive values exits in
            # the pixel neighbordhood, then that pixek is a
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maxium neighborhood
            # difference with the pixel
            
            if z_c:
                if image[i, j] > 0:
                    z_c_image[i,j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
            # Normalize and change datatype to 'uint8' (optional)
            z_c_norm = z_c_image/z_c_image.max()*255
            z_c_image = np.uint8(z_c_norm)
            
            return z_c_image
        
if __name__ == "__main__":
    # Load the image in greyscale
    img = cv2.imread('.//img_test//lena.png')

    img_blur = convolution(img, Mask.LaplacianMask, True, True)
    
    img_log = Zero_crossing(img_blur)
    
    plt.imshow(img_log, cmap='gray')
    plt.title("Log Image")
    plt.show()
    

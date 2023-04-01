# LOG: Laplacian over the Gaussian Filter
# fomular: L(x,y) = Lap2f(x,y) = (delta''f(x,y)/deltax2) + (delta''f(x,y)/deltay2)
# => LoG(x,y) = ([(x2+y2) - 2*sigma^2]/(2*pi*sigma^6))*e^(-(x^2+y^2)/2*sigma^2)
# Steps perform:
# 1. We will define a function for the filter
# 2. Then we will make the mask
# 3. Then will define function to iterate that filter over the image(mask)
# 4. We will make a function for checking the zeros as explained
# 5. And finally a function to bind all this together
# Exam: def L_O_G(x, y, sigma):
    #   nom = (x**2 + y**2) - 2*sigma**2
    #   denom = (2*math.pi*sigma**6)
    #   expo = math.exp(-(x**2 + y**2)/2*sigma**2)
    #   return nom*expo/denom

# Function logic:
# CREATE_LOG function
#   1. Calculate w by using the size of filter and sigma value
#   2. Check if it is even of not, if it even make it odd, by adding 1
#   3. Iterate through the pixels and apply log filter and then append those changed pixels into a new array, and then reshape the array

# CONVOLVE function
#   1. Extract the heights and width of the image
#   2. Now make a range of pixels which are covered by the mask (output of filter)
#   3. Make an array of zeros (res_image)
#   4. Now iterate through the image and append the values of product of mask and the original image and append it to the res_image array

# ZEROS function (to mark the edges):
#   1. Make a zeros array of same dimension of the log_image (zc_image)
#   2. Now iterate over the image and check two values, if they are equal to zero or not. If they are not equal to zero
#      then check if values are positive or negative.
#   3. Now whichever pixels were zero, append those coordinates to zc_image array.
#   4. Make all the pixels in zc_image as 1, meaning white. That will show the edges.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import lance

def L_O_G(x, y, sigma):
    nom = (x**2 + y**2) - 2*sigma**2
    denom = (2*math.pi*sigma**6)
    expo = math.exp(-(x**2 + y**2)/2*sigma**2)
    return nom*expo/denom

def create_log(sigma, size = 7):
    w = math.ceil(float(size)*float(sigma))

    if(w%2 == 0):
        w = w + 1

    l_o_g_mask = []

    w_range = int(math.floor(w/2))
    print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range_inc(-w_range, w_range):
        for j in range_inc(-w_range, w_range):
            l_o_g_mask.append(L_O_G(i,j,sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w,w)
    return l_o_g_mask

def convolve(image, mask):
    width = image.shape[1]
    height = image.shape[0]
    w_range = int(math.floor(mask.shape[0]/2))

    res_image = np.zeros((height, width))

    # Iterate over every pixel that can be covered by the mask
    for i in range(w_range,width-w_range):
        for j in range(w_range,height-w_range):
            # Then convolute with the mask 
            for k in range_inc(-w_range,w_range):
                for h in range_inc(-w_range,w_range):
                    res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]
    return res_image

def z_c_test(l_o_g_image):
    z_c_image = np.zeros(l_o_g_image.shape)

    # Check the sign (negative or positive) of all the pixels around each pixel
    for i in range(1,l_o_g_image.shape[0]-1):
        for j in range(1,l_o_g_image.shape[1]-1):
            neg_count = 0
            pos_count = 0
            for a in range_inc(-1, 1):
                for b in range_inc(-1,1):
                    if(a != 0 and b != 0):
                        if(l_o_g_image[i+a,j+b] < 0):
                            neg_count += 1
                        elif(l_o_g_image[i+a,j+b] > 0):
                            pos_count += 1

            # If all the signs around the pixel are the same and they're not all zero, then it's not a zero crossing and not an edge. 
            # Otherwise, copy it to the edge map.
            z_c = ( (neg_count > 0) and (pos_count > 0) )
            if(z_c):
                z_c_image[i,j] = 1

    return z_c_image
  
def run_l_o_g(bin_image, sigma_val, size_val):
    # Create the l_o_g mask
    print("creating mask")
    l_o_g_mask = create_log(sigma_val, size_val)

    # Smooth the image by convolving with the LoG mask
    print("smoothing")
    l_o_g_image = convolve(bin_image, l_o_g_mask)

    # Display the smoothed imgage
    blurred = plt.add_subplot(1,4,2)
    blurred.imshow(l_o_g_image, cmap='gray')

    # Find the zero crossings
    print("finding zero crossings")
    z_c_image = z_c_test(l_o_g_image)
    print(z_c_image)

    #Display the zero crossings
    edges = plt.add_subplot(1,4,3)
    edges.imshow(z_c_image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    image = cv2.imread(".//img_test//lena.png")
    run_l_o_g(image, 2, 3)
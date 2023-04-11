import cv2
import numpy as np

# Loading source image
src_image = cv2.imread('.//img_test//dog.jpg')
gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

# Creating the kernel(2d convolution matrix)
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
  
# Applying the filter2D() function
img = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel2)
  
# Shoeing the original and output image
cv2.imshow('Original', src_image)
cv2.imshow('Kernel Blur', img)

cv2.waitKey()
cv2.destroyAllWindows()
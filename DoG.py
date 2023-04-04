import cv2
import matplotlib.pyplot as plt


img = cv2.imread('.//img_test//opencv_logo_grey.png')

# Apply 3x3 and 7x7 Gaussian Blur
low_sigma = cv2.GaussianBlur(img, (3, 3), 0)
plt.imshow(low_sigma, cmap='gray')
plt.title('low_sigma')
plt.show()
high_sigma = cv2.GaussianBlur(img, (7, 7), 0)
plt.imshow(high_sigma, cmap='gray')
plt.title('high_sigma')
plt.show()

# Calculate the DoG by subtracting
DoG = low_sigma - high_sigma
plt.imshow(DoG, cmap='gray')
plt.title('DoG')
plt.show()
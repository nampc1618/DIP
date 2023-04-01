import numpy as np

# Mask robert
robertX = np.array([[-1, 0],
                    [0, 1]])
print(robertX)
robertY = np.array([[0, -1],
                    [1, 0]])
print(robertY)

# Mask Prewitt
prewittX = np.array([[-1, -1 , -1],
                     [0, 0, 0],
                     [1, 1, 1]])
prewittY = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
print(prewittX)
print(prewittY)

# Mask Sobel
sobelH = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobelV = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

print(sobelH)
print(sobelV)

# 5x5 LoG filter
LoGfilter55 = np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]])
print(LoGfilter55)

# 17x17 LoGfilter
LoGfilter1717 = np.array([[0, 0, 0, 0, 0, 0, - 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, - 1, - 1, - 1, -1, -1, -1, -1, - 1, - 1, 0, 0, 0, 0],
                        [0, 0, - 1, - 1, -1, -2, -3, -3, -3, -3, -3, -2, -1, - 1, - 1, 0, 0],
                        [0, 0, - 1, - 1, -2, -3, -3, -3, -3, -3, -3, -3, -2, - 1, - 1, 0, 0],
                        [0, -1, -1, -2, -3, -3, -3, -2, -3, -2, -3, -3, -3, -2, -1, -1, 0],
                        [0, -1, -2, -3, -3, -3, 0, 2, 4, 2, 0, -3, -3, -3, -2, -1, 0],
                        [ -1, -1, -3, -3, -3, 0, 4, 10, 12, 10, 4, 0, -3, -3, -3, -1, -1],
                        [-1, -1, -3, -3, -2, 2, 10, 18, 21, 18, 10, 2, -2, -3, -3, -1, -1],
                        [-1, -1, -3, -3, -3, 4, 12, 21, 24, 21, 12, 4, -3, -3, -3, -1, -1],
                        [-1, -1, -3, -3, -2, 2, 10, 18, 21, 18, 10, 2, -2, -3, -3, -1, -1],
                        [ -1, -1, -3, -3, -3, 0, 4, 10, 12, 10, 4, 0, -3, -3, -3, -1, -1],
                        [0, -1, -2, -3, -3, -3, 0, 2, 4, 2, 0, -3, -3, -3, -2, -1, 0],
                        [0, -1, -1, -2, -3, -3, -3, -2, -3, -2, -3, -3, -3, -2, -1, -1, 0],
                        [0, -1, -1, -2, -3, -3, -3, -2, -3, -2, -3, -3, -3, -2, -1, -1, 0],
                        [0, 0, - 1, - 1, -1, -2, -3, -3, -3, -3, -3, -2, -1, - 1, - 1, 0, 0],
                        [0, 0, 0, 0, - 1, - 1, - 1, -1, -1, -1, -1, - 1, - 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, - 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0]])
print(LoGfilter1717)
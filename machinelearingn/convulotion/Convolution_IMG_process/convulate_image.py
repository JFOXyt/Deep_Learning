import numpy as np
from PIL import Image
import sys
from collections import deque
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

KERNEL_X = 3
PIX_K = 1/9
a = int(input("1/9 KERNEL [1] \nGaussian KERNEL [2]\nSame [3]\nHorizontal Edge KERNEL[4]\nVertical Edge KERNEL[5]\nSharpening KERNEL[6]\nDouble Image KERNEL[7]\nChoose Option: "))


if a == 1:
    kernel = np.tile(PIX_K, KERNEL_X**2)
elif a == 2:
    kernel = np.array([1/16,1/8,1/16
                    ,1/8,1/4,1/8
                    ,1/16,1/8,1/16])
elif a == 3:
    kernel = np.array([0,0,0,
                        0,1,0,
                        0,0,0])
elif a == 4:
    kernel = np.array([-1,-1,-1,
                        0,0,0,
                        1,1,1])
elif a == 5:
    kernel = np.array([-1,0,1,
                       -1,0,1,
                       -1,0,1])
elif a == 6:
    kernel = np.array([0,-1,0,
                       -1,5,-1,
                       0,-1,0])
else:
    kernel = np.array([1/2,0,0,
                       0,0,0,
                       0,0,1/2])

im = Image.open('convulotion/Convolution_IMG_process/cnn_img.png')
rgb_im = im.convert('RGB')

width, height = im.size

lists = deque()
for piy in range(height):
    for pix in range(width):

        r, g, b = rgb_im.getpixel((pix, piy))
        lists.append([r,g,b]) 

pixel_color = np.array(lists)

pix_l = []

result = convolve(pixel_color,kernel[:, np.newaxis])
result = np.reshape(result, (width,height,3))

plt.imshow(result)
plt.show()



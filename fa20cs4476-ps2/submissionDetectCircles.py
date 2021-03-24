import numpy as np
import math
import imageio
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.feature import canny
import matplotlib.pyplot as plt
from typing import Tuple

def detect_circles(img, radius, use_gradient):
    image = rgb2gray(img).astype(np.float64)
    edges = canny(image, sigma = 1)
    gradient = np.gradient(image)
    gradientX = gradient[1]
    gradientX[gradientX == 0] = 0.000000001
    gradientY = gradient[0]
    gradient_direction = np.arctan2(gradientY, gradientX)
    edgeRow, edgeCol = edges.shape
    hough = np.zeros((edgeRow // 2, edgeCol // 2))
    for y in range(0, edgeRow):
        for x in range(0, edgeCol):
            if edges[y, x]:
                if not use_gradient:
                    for theta in np.arange(0, 2 * math.pi, .05):
                        a = np.around(x + radius * np.cos(theta)).astype(int)
                        b = np.around(y + radius * np.sin(theta)).astype(int)
                        if a >= 0 and b >= 0 and a < edgeRow and b < edgeCol:
                            hough[a // 2, b // 2] += 1
                else:
                    theta = gradient_direction[y, x]
                    a = np.around(x + radius * np.cos(theta)).astype(int)
                    b = np.around(y + radius * np.sin(theta)).astype(int)
                    if a >= 0 and b >= 0 and a < edgeRow and b < edgeCol:
                        hough[a // 2, b // 2] += 1
                    a = np.around(x - radius * np.cos(theta)).astype(int)
                    b = np.around(y - radius * np.sin(theta)).astype(int)
                    if a >= 0 and b >= 0 and a < edgeRow and b < edgeCol:
                        hough[a // 2, b // 2] += 1
    plt.imshow(hough)
    #imageio.imwrite('egg_gradient_false_accumulator.jpg', hough.astype(np.uint8))
    plt.show()
    return np.argwhere(hough == np.amax(hough))

image = imageio.imread('egg.jpg')
radius = 5
centers = detect_circles(image, radius, False)
print(centers)
imgRow, imgCol, color = image.shape
for i in np.arange(0, len(centers[:, 0])):
    for theta in np.arange(0, 2 * math.pi, 0.01):
        a = np.around(centers[i, 0] + radius * np.cos(theta)).astype(int)
        b = np.around(centers[i, 1] + radius * np.sin(theta)).astype(int)
        if a >= 0 and b >= 0 and a < imgCol and b < imgRow:
            image[b, a] = [255, 0, 0]
plt.imshow(image)
#imageio.imwrite('egg_gradient_6.jpg', image)
plt.show()


#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# Edit SeamCarving.ipynb instead.
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from imageio import imread, imsave

def  rgb2gray(img):
    """
    Converts an RGB image into a greyscale image

    Input: ndarray of an RGB image of shape (H x W x 3)
    Output: ndarray of the corresponding grayscale image of shape (H x W)

    """

    if(img.ndim != 3 or img.shape[-1] != 3):
        print("Invalid image! Please provide an RGB image of the shape (H x W x 3) instead.".format(img.ndim))
        return None

    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def compute_gradients(img):
    """
    Computes the gradients of the input image in the x and y direction using a
    differentiation filter.

    ##########################################################################
    # TODO: Design a differentiation filter and update the docstring. Stick  #
    # to a pure differentiation filter for this assignment.                  #
    # Hint: Look at Slide 14 from Lecture 3: Gradients.                      #
    ##########################################################################

    Input: Grayscale image of shape (H x W)
    Outputs: gx, gy gradients in x and y directions respectively

    """
    gx = gy = np.zeros_like(img)

    diffx = np.array(
        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    )

    diffy = np.array(
        [[-1, -1, -1],
         [0, 0, 0],
         [1, 1, 1]]
    )

    diffx = np.flipud(diffx)
    diffx = np.fliplr(diffx)

    diffy = np.fliplr(diffy)
    diffy = np.flipud(diffy)

    gx = ndimage.correlate(img, diffx, mode = 'constant').astype(np.float64)
    gy = ndimage.correlate(img, diffy, mode = 'constant').astype(np.float64)

    ##########################################################################
    # TODO: Design a pure differentiation filter and use correlation to      #
    # compute the gradients gx and gy. You might have to try multiple        #
    # filters till the test below passes. All the tests after will fail if   #
    # this one does not pass.                                                #
    ##########################################################################
    return gx, gy

def energy_image(img):
    """
    Computes the energy of the input image according to the energy function:

        e(I) = abs(dI/dx) + abs(dI/dy)

    Use compute_gradients() to help you calculate the energy image. Remember to normalize
    energyImage by dividing it by max(energyImage).

    Input: image of the form (H x W) or (H x w x 3)
    Output: array of energy values of the image computed according to the energy function.

    """
    if(img.ndim == 3 and img.shape[-1] == 3):
        img = rgb2gray(img)

    energyImage = np.zeros_like(img)

    gx, gy = compute_gradients(img)

    energyImage = (np.abs(gx) + np.abs(gy))

    energyImage = (energyImage / np.amax(energyImage)).astype(np.float64)

    ##########################################################################
    # TODO: Compute the energy of input using the defined energy function.   #                                             #
    ##########################################################################

    return energyImage

def cumulative_minimum_energy_map(energyImage, seamDirection):
    """
    Calculates the cumulative minim energy map according to the function:

        M(i, j) = e(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))

    Inputs:
        energyImage: Results of passign the input image to energy_image()
        seamDirection: 'HORIZONTAL' or 'VERTICAL'

    Output: cumulativeEnergyMap

    """

    cumulativeEnergyMap = np.zeros_like(energyImage)
    if seamDirection == "VERTICAL":
        cumulativeEnergyMap[0] = energyImage[0]
        for i in range(1, cumulativeEnergyMap.shape[0]):
            cumulativeEnergyMap[i] = energyImage[i] + ndimage.minimum_filter(cumulativeEnergyMap[i - 1], footprint = np.ones(3))
    elif seamDirection == "HORIZONTAL":
        cumulativeEnergyMap[:,0] = energyImage[:,0]
        for j in range(1, cumulativeEnergyMap.shape[1]):
            cumulativeEnergyMap[:,j] = energyImage[:,j] + ndimage.minimum_filter(cumulativeEnergyMap[:, j - 1], footprint = np.ones(3))

    cumulativeEnergyMap = cumulativeEnergyMap.astype(np.float64)
    ##########################################################################
    # TODO: Compute the cumulative minimum energy map in the input           #
    # seamDirection for the input energyImage. It is fine it is not fully    #
    # vectorized.                                                            #
    ##########################################################################

    return cumulativeEnergyMap

def find_optimal_vertical_seam(cumulativeEnergyMap):
    """
    Finds the least connected vertical seam using a vertical cumulative minimum energy map.

    Input: Vertical cumulative minimum energy map.
    Output:
        verticalSeam: vector containing column indices of the pixels in making up the seam.

    """

    verticalSeam = [0]*cumulativeEnergyMap.shape[0]
    verticalSeam[cumulativeEnergyMap.shape[0] - 1] = np.where(cumulativeEnergyMap[cumulativeEnergyMap.shape[0] - 1] == np.amin(cumulativeEnergyMap[cumulativeEnergyMap.shape[0] - 1]))[0].astype(np.float64)[0]
    for i in range(cumulativeEnergyMap.shape[0] - 2, -1, -1):
        arr = np.where(cumulativeEnergyMap[i] == ndimage.minimum_filter(cumulativeEnergyMap[i], footprint=np.ones(3))[verticalSeam[i + 1].astype(int)])[0].astype(np.float64)
        arr = arr[(arr >= verticalSeam[i + 1] - 1) & (arr <= verticalSeam[i + 1] + 1)]
        verticalSeam[i] = arr[0]

    ##########################################################################
    # TODO: Find the minimal connected vertical seam using the input         #
    # cumulative minimum energy map.                                         #
    ##########################################################################

    return verticalSeam

def find_optimal_horizontal_seam(cumulativeEnergyMap):
    """
    Finds the least connected horizontal seam using a horizontal cumulative minimum energy map.

    Input: Horizontal cumulative minimum energy map.
    Output:
        horizontalSeam: vector containing row indices of the pixels in making up the seam.

    """
    horizontalSeam = [0]*cumulativeEnergyMap.shape[1]
    transpose = np.transpose(cumulativeEnergyMap)
    horizontalSeam = find_optimal_vertical_seam(transpose)

    ##########################################################################
    # TODO: Find the minimal connected horizontal seam using the input       #
    # cumulative minimum energy map.                                         #
    ##########################################################################

    return horizontalSeam

def reduce_width(img, energyImage):
    """
    Removes pixels along a seam, reducing the width of the input image by 1 pixel.

    Inputs:
        img: RGB image of shape (H x W x 3) from which a seam is to be removed.
        energyImage: The energy image of the input image.

    Outputs:
        reducedColorImage: The input image whose width has been reduced by 1 pixel
        reducedEnergyImage: The energy image whose width has been reduced by 1 pixel
    """
    reducedEnergyImageSize = (energyImage.shape[0], energyImage.shape[1] - 1)
    reducedColorImageSize = (img.shape[0], img.shape[1] - 1, 3)
    print(reducedEnergyImageSize, reducedColorImageSize)

    reducedColorImage = np.zeros(reducedColorImageSize)
    reducedEnergyImage = np.zeros(reducedEnergyImageSize)


    energyMap = cumulative_minimum_energy_map(energyImage, "VERTICAL")
    energySeam = find_optimal_vertical_seam(energyMap)

    for i in range(0, reducedEnergyImageSize[0]):
        reducedEnergyImage[i] = np.delete(energyImage[i], energySeam[i].astype(int))

    for j in range(0, reducedColorImageSize[0]):
        reducedColorImage[j] = np.delete(img[j], energySeam[j].astype(int), 0)
    ##########################################################################
    # TODO: Compute the cumulative minimum energy map and find the minimal   #
    # connected vertical seam. Then, remove the pixels along this seam.      #
    ##########################################################################

    return reducedColorImage, reducedEnergyImage

def reduce_height(img, energyImage):
    """
    Removes pixels along a seam, reducing the height of the input image by 1 pixel.

    Inputs:
        img: RGB image of shape (H x W x 3) from which a seam is to be removed.
        energyImage: The energy image of the input image.

    Outputs:
        reducedColorImage: The input image whose height has been reduced by 1 pixel
        reducedEnergyImage: The energy image whose height has been reduced by 1 pixel
    """

    reducedEnergyImageSize = tuple((energyImage.shape[0] - 1, energyImage.shape[1]))
    reducedColorImageSize = tuple((img.shape[0] - 1, img.shape[1], 3))

    reducedColorImage = np.zeros(reducedColorImageSize)
    reducedEnergyImage = np.zeros(reducedEnergyImageSize)

    colorInputTranspose = np.transpose(img, (1, 0, 2))
    energyInputTranspose = np.transpose(energyImage)

    reducedColorTrans, reducedEnergyTrans = reduce_width(colorInputTranspose, energyInputTranspose)

    reducedColorImage = np.transpose(reducedColorTrans, (1, 0, 2))
    reducedEnergyImage = np.transpose(reducedEnergyTrans)

    ##########################################################################
    # TODO: Compute the cumulative minimum energy map and find the minimal   #
    # connected horizontal seam. Then, remove the pixels along this seam.    #
    ##########################################################################

    return reducedColorImage, reducedEnergyImage

def seam_carving_reduce_width(img, reduceBy):
    """
    Reduces the width of the input image by the number pixels passed in reduceBy.

    Inputs:
        img: Input image of shape (H x W X 3)
        reduceBy: Positive non-zero integer indicating the number of pixels the width
        should be reduced by.

    Output:
        reducedColorImage: The result of removing reduceBy number of vertical seams.
    """

    #reducedColorImage = img[:, reduceBy//2:-reduceBy//2, :]  #crops the image
    imgCopy = np.copy(img)

    for i in range(0, reduceBy):
        reducedColorImageSize = (imgCopy.shape[0], imgCopy.shape[1] - 1, 3)
        reducedColorImage = np.zeros(reducedColorImageSize)

        imgEnergy = energy_image(imgCopy)
        imgMap = cumulative_minimum_energy_map(imgEnergy, "VERTICAL")
        imgSeam = find_optimal_vertical_seam(imgMap)

        for j in range(0, reducedColorImageSize[0]):
            reducedColorImage[j] = np.delete(imgCopy[j], imgSeam[j].astype(int), 0)

        imgCopy = reducedColorImage

    reducedColorImage = reducedColorImage.astype(np.uint8)
    ##########################################################################
    # TODO: For the Prague image, write a few lines of code to call the      #
    # we have written to find and remove 100 vertical seams                  #
    ##########################################################################

    return reducedColorImage

def seam_carving_reduce_height(img, reduceBy):
    """
    Reduces the height of the input image by the number pixels passed in reduceBy.

    Inputs:
        img: Input image of shape (H x W X 3)
        reduceBy: Positive non-zero integer indicating the number of pixels the
        height should be reduced by.

    Output:
        reducedColorImage: The result of removing reduceBy number of horizontal
        seams.
    """

    #reducedColorImage = img[reduceBy//2:-reduceBy//2, :, :]  #crops the image
    imgTranspose = np.transpose(img, (1, 0, 2))
    reducedColorImage = seam_carving_reduce_width(imgTranspose, reduceBy)
    reducedColorImage = np.transpose(reducedColorImage, (1, 0, 2))
    ##########################################################################
    # TODO: For the Prague image, write a few lines of code to call the      #
    # we have written to find and remove 100 horizontal seams.               #
    ##########################################################################

    return reducedColorImage
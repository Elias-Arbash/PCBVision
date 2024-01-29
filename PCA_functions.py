from hylite import HyImage #, HyData
from hylite.filter.dimension_reduction import PCA #, from_loadings
from tqdm import tqdm
from time import sleep
import numpy as np
import cv2
import torch


### Functions for HSI_PCA_data_generation ###

def PCA2(x, c):
    """
    Perform Principal Component Analysis (PCA) on the input HyImage object.

    Parameters:
        x (HyImage): The HyImage object containing the data to be analyzed.
        c (list): The list of bands to be used for the PCA analysis.

    Returns:
        HyImage: The HyImage object containing the PCA-transformed data.
    """
    
    # Convert the input data to a HyImage object
    x = HyImage(x)  
    
    # Perform PCA on the HyImage object using the specified bands
    pca, loadings, means = PCA(x, bands=c)  

    # Extract the PCA-transformed data from the PCA object
    x = pca.data  
    
    # Extract the wavelengths corresponding to the PCA components
    v = pca.get_wavelengths()  

    # Return the PCA-transformed data, wavelengths, loadings, and means
#     return x, v, loadings, means

    return x  # Only return the PCA-transformed data for simplicity

def clipping_neg_pos(hscubes):
    """
    Clip all negative and positive values in the HSCubes array to 0 and 1, respectively.

    Parameters:
        hscubes (numpy.ndarray): PCB-Vision HS cubes array.

    Returns:
        clipped (numpy.ndarray): Clipped HSCubes array.
    """

    clipped = []

    # Iterate through each HSCube in the array
    for i in tqdm(range(len(hscubes))):
        sleep(0.001)  # Introduce a small delay to prevent excessive memory consumption

        # Copy the current HSCube
        img = hscubes[i].copy()

        # Clip negative values to 0
        img[np.where(img < 0.0)] = 0.0  # Replace negative values with 0

        # Clip positive values to 1
        img[np.where(img > 1.0)] = 1.0  # Replace positive values with 1

        # Reshape the clipped HSCube
        img = img[:, :, :]

        # Append the clipped HSCube to the list of clipped cubes
        clipped.append(img)

        # Delete the copied HSCube to conserve memory
        del img

    print("Clipping data is complete. No more negative values.")
    return clipped

def slicing(hscubes, x):
    """
    Slices the given HSCubes array, removing the first `x` bands (channels) from each HSCube.

    Parameters:
        hscubes (numpy.ndarray): HSCubes array containing 224 bands (channels).
        x (int): Number of bands (channels) to remove from the beginning of each HSCube.

    Returns:
        sliced (numpy.ndarray): Sliced HSCubes array
    """

    sliced = []

    # Iterate through each HSCube in the array
    for i in tqdm(range(len(hscubes))):
        sleep(0.001)  # Introduce a small delay to prevent excessive memory consumption

        # Copy the current HSCube
        img = hscubes[i].copy()

        # Slice out the first `x` bands (channels)
        img = img[:, :, x:]

        # Append the sliced HSCube to the list of sliced cubes
        sliced.append(img)

        # Delete the copied HSCube to conserve memory
        del img

    print("Skipping first " + str(x) + " bands")
    print("The HS cubes have " + str(224 - x) + " channels, the first " + str(x) + " are sliced out.")

    return sliced

### Functions for HSI_PCA ###

def resize_hyperspectral_images(HSI_list, Mask_list, resolution=640):
    """
    Resizes a list of hyperspectral images and their corresponding segmentation masks 
    to the specified resolution.

    Parameters:
        HSI_list (list): List of hyperspectral images (numpy.ndarray)
        Mask_list (list): List of segmentation masks (numpy.ndarray)
        resolution (int, optional): Desired resolution for resized images (default: 640)

    Returns:
        list: Resized hyperspectral image list
        list: Resized segmentation mask list
    """

    # Initialize list to store resized hyperspectral images
    resized_HSI_list = []  
    # Initialize list to store resized segmentation masks
    resized_Mask_list = []  

    # Define target resolution
    target_size = (resolution, resolution)  

    for HSI, mask in zip(HSI_list, Mask_list):
        """
        Iterate over hyperspectral images and their corresponding masks
        """

        # Resize hyperspectral image
        resized_HSI = cv2.resize(HSI, target_size)

        # Resize segmentation mask using nearest neighbor interpolation
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Append resized images and masks to their respective lists
        resized_HSI_list.append(resized_HSI)
        resized_Mask_list.append(resized_mask)

    return resized_HSI_list, resized_Mask_list

def resize_segmentation_masks(mask, new_shape):
    """
    Resize a list of segmentation masks while preserving class values to the specified 
    new shape (row x col).

    Parameters:
        masks (list of numpy arrays): A list of segmentation masks with values 
        [0, 1, 2, 3].
        new_shape (tuple): A tuple representing the new shape (row, col) to which the
          masks should be resized.

    Returns:
        list of numpy arrays: A list of resized segmentation masks with preserved class 
        values.
    """
    
    # Resize the mask to the new shape using OpenCV with nearest-neighbor interpolation
    resized_mask = cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure the data type is integer
    resized_mask = resized_mask.astype(np.int)

    # Clip values to ensure they are in the [0, 3] range
    resized_mask = np.clip(resized_mask, 0, 3)

    return resized_mask
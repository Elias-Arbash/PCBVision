# Importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spectral as spi
from scipy import io
import cv2
import sys
import matplotlib as mpl
import seaborn as sns
import timeit
import logging
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from time import sleep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerBase
from spectral import *

logger = logging.getLogger(__name__)
logger.info('Elias Arbash: HSI/RGB PCBs Benchmark Dataset Functions')

def visualize(mask):
    """
    Show a PCB-vision mask in using specific colormap.
    
    Parameters:
        mask (numpy.ndarray): the 2D mask image.
    """
    
    colours = ['black','red', 'green', 'blue', 'Yellow']
    classes = {0:'black', 1:'red', 2:'green', 3:'blue', 4:'Yellow'}
    cmap = []
    for i,x in enumerate(np.unique(mask)):
        cmap.append(classes[x])
    cmap = mpl.colors.ListedColormap(cmap)
    colormap = plt.imshow(mask, cmap=cmap, interpolation="none")
    plt.axis('off')

def read_hsi_mask(datapath, GTpath):
    """
    Reads an HSI and its corresponding mask from specified paths and returns the mask as a NumPy array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        GTpath (str): Path to the corresponding mask file.

    Returns:
        mask (numpy.ndarray): NumPy array containing the loaded mask.
    """
    # Convert the GTpath to POSIX format for compatibility
    GTpath2 = GTpath.as_posix()

    # Strip the extension
    spectral_file = str(GTpath2[:-4])  

    # Open the HSI data using ENVI and access the mask band using bipfile
    numpy_ndarr = envi.open(GTpath, spectral_file)
    y = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)

    mask = y[:, :, 0]

    return mask

def read_hsi_cube(datapath, Cubepath):
    """
    Reads an HSI cube from the specified path and returns it as a NumPy array.

    Parameters:
        datapath (str): Path to the directory containing the HSI data.
        Cubepath (str): Path to the HSI cube file.

    Returns:
        hsi_cube (numpy.ndarray): NumPy array containing the loaded HSI cube.
    """

    # Construct the paths to the header and spectral files
    header_file = str(datapath / Cubepath)
    # Remove extension
    spectral_file = str(datapath / Cubepath[:-4])  

    # Open the HSI data using ENVI and access the entire cube using bipfile
    numpy_ndarr = envi.open(header_file, spectral_file)
    hsi_cube = spi.io.bipfile.BipFile.open_memmap(numpy_ndarr)

    # Return the loaded HSI cube as a NumPy array
    return hsi_cube

    
def read_dataset(dataset_path):
    """This function reads the dataset.
    
    Parameters:
        - dataset_path (str): The path of the dataset folder.

    Returns:
        A tuple of 7 lists of 53 elements:
        
        - Hyperspectral Images (HSI) of the 53 PCB
        - HSI general segmentation masks
        - HSI mono segmentation masks
        - RGB images of the 53 PCB
        - RGB general segmentation masks
        - RGB mono segmentation masks
        - PCB Masks of the HSI
    """
    hsi_path = dataset_path + 'HSI/'
    rgb_path = dataset_path + 'RGB/'
    
    HSI = []
    HSI_seg_masks = []
    HSI_mono_masks = []
    RGB = []
    RGB_mono_masks = []
    RGB_general_masks = []
    PCB_Masks = []
    
    for i in tqdm(range(1,54)):
        sleep(0.001)
        datapath = Path( hsi_path + 'pcb' + str(i))
        Cubepath = "pcb" + str(i) + ".hdr"
        
        Maskpath = Path(hsi_path + 'General_masks/' + str(i) + ".HDR")
        HSI.append(read_hsi_cube(datapath, Cubepath))
        HSI_seg_masks.append(read_hsi_mask(datapath, Maskpath))
    
        Maskpath = Path(hsi_path + 'Monoseg_masks/mono' + str(i) + ".hdr")
        HSI_mono_masks.append(read_hsi_mask(datapath, Maskpath))
    
        RGBpath = rgb_path + str(i) + '.jpg'
        RGB.append(cv2.cvtColor(cv2.imread(RGBpath),cv2.COLOR_BGR2RGB))
    
        RGB_mono_masks_path = rgb_path + 'Monoseg/' + str(i) + '.png'
        RGB_mono_masks.append(np.array(Image.open(RGB_mono_masks_path)))
    
        RGB_general_masks_path = rgb_path + 'General/' + str(i) + '.png'
        RGB_general_masks.append(np.array(Image.open(RGB_general_masks_path)))
    
        maskspath = Path(hsi_path + 'PCB_Masks/'+str(i) + ".jpg")
        PCB_Masks.append(cv2.cvtColor(cv2.imread(str(maskspath) ),cv2.COLOR_BGR2GRAY))

    print("Dataset loading is complete.")
    return HSI, HSI_seg_masks, HSI_mono_masks, RGB, RGB_mono_masks, RGB_general_masks, PCB_Masks



# Function to train a model on a specific GPU
def set_gpu(gpu_id):
    """
    Function to set the current device to a specific GPU

    Parameters:
        gpu_id (int): The ID of the GPU to use

    Returns:
        torch.device: The selected GPU device
    """

    # Set the current device to the specified GPU ID
    torch.cuda.set_device(gpu_id)
    
    # Return the selected GPU device
    return torch.device("cuda")


def Generate_Training_data(training_list, HSI_cubes, seg_masks):
    """
    Generates augmented training data for the given HS cubes and their
    corresponding masks. The function for reading PCB-Vision HS
    training cubes and generating the training set.
    
    Parameters:
        training_list (list): A list of indices corresponding to PCB-Vision 
                              HS cubes and masks to be augmented.        
        HSI_cubes (list): A list of HS data cubes
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): A list of the augmented HS cubes 
        masks (list): A list of the augmented masks
        
    """
    cubes = []
    masks = []
    for i, ii in enumerate(training_list):
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        cube_aug, masks_aug = data_augmentation(HSI_cubes[ii-1], seg_masks[ii-1])
        for j in range(len(cube_aug)):
            cubes.append(cube_aug[j])
            masks.append(masks_aug[j])
        
        del cube_aug, masks_aug
        
    return cubes, masks

def Generate_data(data_list, HSI_cubes, seg_masks):
    """
    Reading PCB-Vision validation and testing HS cubes.
    It does not perform any augmentation
    
    Args:
        data_list (list): A list of indices corresponding to the HS cubes in 
                          PCB-Vision to be read.
        HSI_cubes (list): A list of the augmented HS cubes 
        seg_masks (list): A list of the ground truth masks
        
    Returns:
        cubes (list): HS cubes
        masks (list): segmentation masks
    
    """
    cubes = []
    masks = []
    for i, ii in enumerate(data_list):
        cubes.append(HSI_cubes[ii-1])
        masks.append(seg_masks[ii-1])
        
    return cubes, masks

def data_augmentation(hsi_cube, mask):
    np.random.seed(0)
    """
    Augments a hyperspectral cube and its corresponding mask with different transformations.

    Args:
        hsi_cube (ndarray): A hyperspectral cube of shape (rows, columns, bands).
        mask (ndarray): A 2D mask of shape (rows, columns).

    Returns:
        A tuple of two lists:
        - The first list contains 7 augmented hyperspectral cubes, each of shape (rows, columns, bands).
        - The second list contains 7 augmented masks, each of shape (rows, columns).
    """
    # Initialize empty lists to store the augmented cubes and masks
    augmented_cubes = []
    augmented_masks = []

    # Get the number of rows and columns of the mask
    rows, cols = mask.shape
    #rows, cols = hsi_cube.shape[:2]

    # Random rotation clockwise
    angle = 15#np.random.randint(1, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    augmented_cubes.append(rotated_hsi)
    augmented_masks.append(rotated_mask)

    # Random rotation counter clockwise
    angle = 15#np.random.randint(1, 20)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -angle, 1)
    rotated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
    augmented_cubes.append(rotated_hsi)
    augmented_masks.append(rotated_mask)

    # Random vertical translation
    if np.random.rand() < 0.5:
        v_trans = np.random.randint(-30, -9)
    else:
        v_trans = np.random.randint(10, 31)
    M = np.float32([[1, 0, 0], [0, 1, v_trans]])
    translated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    translated_mask = cv2.warpAffine(mask, M, (cols, rows),flags=cv2.INTER_NEAREST)
    augmented_cubes.append(translated_hsi)
    augmented_masks.append(translated_mask)

    # Random horizontal translation
    if np.random.rand() < 0.5:
        h_trans = np.random.randint(-30, -9)
    else:
        h_trans = np.random.randint(10, 31)
    M = np.float32([[1, 0, h_trans], [0, 1, 0]])
    translated_hsi = cv2.warpAffine(hsi_cube, M, (cols, rows))
    translated_mask = cv2.warpAffine(mask, M, (cols, rows))
    augmented_cubes.append(translated_hsi)
    augmented_masks.append(translated_mask)

    # Flip on the vertical axis
    flipped_hsi = np.flip(hsi_cube, axis=0)
    flipped_mask = np.flip(mask, axis=0)
    augmented_cubes.append(flipped_hsi)
    augmented_masks.append(flipped_mask)

    # Flip on the ç axis
    flipped_hsi = np.flip(hsi_cube, axis=1)
    flipped_mask = np.flip(mask, axis=1)
    augmented_cubes.append(flipped_hsi)
    augmented_masks.append(flipped_mask)

    return augmented_cubes, augmented_masks


def evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes): # Yes Please
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    # Calculate pixel accuracy per class
    pixel_accuracy_per_class = true_positive_sum / (true_positive_sum + false_negative_sum)

    # Calculate pixel accuracy
    pixel_accuracy = np.sum(true_positive_sum) / np.sum(confusion_matrix_sum)

    # Calculate precision, recall, F1 score
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)
    recall = true_positive_sum / (true_positive_sum + false_negative_sum)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Calculate Intersection Over Union (IoU)
    iou = intersection_sum / union_sum

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_sum) / (np.sum(confusion_matrix_sum, axis=1) + np.sum(confusion_matrix_sum, axis=0))

    # Calculate Kappa coefficient
    total_pixels = np.sum(confusion_matrix_sum)
    observed_accuracy = np.sum(true_positive_sum) / total_pixels
    expected_accuracy = np.sum(true_positive_sum) * np.sum(confusion_matrix_sum, axis=1) / total_pixels**2
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    # Return the calculated evaluation metrics
    return confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum, precision, recall, f1_score, pixel_accuracy_per_class,   pixel_accuracy, iou, dice_coefficient, kappa

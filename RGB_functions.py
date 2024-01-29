import numpy as np
from PIL import Image
import torch
import cv2

def calculate_mean_std(image_list, target_resolution=(640, 640)):
    """
    Calculate mean and standard deviation across all images and channels.

    Parameters:
        image_list (list of numpy.ndarray): List of input images.
        target_resolution (tuple): Target resolution for resizing images.

    Returns:
        tuple: Mean and standard deviation values for each channel.
    """
    total_pixels = 0
    channel_sums = [0, 0, 0]  # Initialize channel sums

    for image in image_list:
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        # Resize image to the target resolution
        image = image.resize(target_resolution, Image.BILINEAR)

        # Convert PIL Image back to NumPy array
        image = np.array(image)

        # Update channel sums
        channel_sums[0] += np.sum(image[:, :, 0])
        channel_sums[1] += np.sum(image[:, :, 1])
        channel_sums[2] += np.sum(image[:, :, 2])

        # Update total number of pixels
        total_pixels += np.prod(image.shape[:2])

    # Calculate mean values for each channel
    channel_means = [sums / total_pixels for sums in channel_sums]

    # Calculate standard deviation values for each channel
    channel_variances = [np.sum((image[:, :, i] - channel_means[i]) ** 2) for i in range(3)]
    channel_stds = [np.sqrt(variance / total_pixels) for variance in channel_variances]

    return tuple(channel_means), tuple(channel_stds)

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

def resize_segmentation_masks(mask, new_shape):
    """
    Resize a list of segmentation masks while preserving class values to the 
    specified new shape (row x col).

    Parameters:
        masks (list of numpy arrays): A list of segmentation masks with values [0, 1, 2, 3].
        new_shape (tuple): A tuple representing the new shape (row, col) to which the masks
          should be resized.

    Returns:
        list of numpy arrays: A list of resized segmentation masks with preserved class values.
    """
    
    # Resize the mask to the new shape using OpenCV with nearest-neighbor interpolation
    resized_mask = cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure the data type is integer
    resized_mask = resized_mask.astype(np.int)

    # Clip values to ensure they are in the [0, 3] range
    resized_mask = np.clip(resized_mask, 0, 3)

    return resized_mask
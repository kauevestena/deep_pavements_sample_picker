
from libs.mapillary_funcs import *
import numpy as np
from cmocean.cm import phase
from scipy import ndimage


def get_discrete_colormap(num_steps):
  """
  Generates a discretized colormap list from the "phase" colormap in "cmocean".

  Args:
      num_steps: The desired number of discrete steps in the colormap (excluding endpoints).

  Returns:
      A list of RGB color values representing the discretized colormap.
  """

  cmap = phase  # Choose the desired colormap from cmocean

  # Generate positions excluding endpoints for unique colors
  positions = np.linspace(0.01, 0.99, num_steps, endpoint=False)

  # Get RGB values from the colormap at the positions
  colors = cmap(positions)[:, :3]  # Extract RGB values (excluding alpha)

  # Convert colors from [0, 1] range to [0, 255] for integer representation
  colors = (colors * 255).astype(np.uint8)

  # Return the list of discretized color values
  return colors.tolist()

import cv2
import numpy as np

def combine_segmentation(image_path, mask_dict, color_dict,outpath=None):
  """
  Combines an image with multiple binary masks and assigns colors based on a color dictionary.

  Args:
    image_path: Path to the original image.
    mask_dict: Dictionary containing lists of paths to masks for each object class.
    color_dict: Dictionary mapping object classes to their corresponding RGB colors.

  Returns:
    A segmented image as a NumPy array.
  """
  # Load the original image
  image = cv2.imread(image_path)

  # Initialize an empty image for segmentation
  segmented_image = np.zeros_like(image, dtype=np.uint8)

  # Loop through each object class and its corresponding masks
  for class_name, mask_paths in mask_dict.items():
    # Combine all masks for the current class (logical OR)
    combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    for mask_path in mask_paths:
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Get the corresponding color for the class
    color = color_dict.get(class_name, None)

    # Check if color is defined and apply it to the segmented image based on the mask
    if color is not None:
      segmented_image[combined_mask > 0] = color

  if outpath:
    cv2.imwrite(outpath, segmented_image)

  return segmented_image

def blend_images(image1, image2, alpha=0.5,outpath=None):
  """
  Blends two images with a specified alpha value.

  Args:
    image1: First image as a NumPy array.
    image2: Second image as a NumPy array.
    alpha: Weighting factor for blending (0.0 to 1.0).

  Returns:
    The blended image as a NumPy array.
  """
  # if the images are paths, read them:
  if isinstance(image1, str):
    image1 = cv2.imread(image1)

  if isinstance(image2, str):
    image2 = cv2.imread(image2)

  # Ensure images have the same shape
  if image1.shape != image2.shape:
    raise ValueError("Images must have the same shape for blending.")

  # Blend the images using weighted addition
  blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

  if outpath:
    cv2.imwrite(outpath, blended_image)

  return blended_image

def create_rgb_matrix(matrix, colordict):
    """
    Create an RGB matrix from the input matrix using the provided colordict.

    Args:
        matrix (np.ndarray): Input numpy matrix.
        colordict (dict): Dictionary mapping keys to RGB color values.

    Returns:
        np.ndarray: RGB matrix.
    """
    # Create an empty result matrix with shape (height, width, 3)
    h, w = matrix.shape
    result_matrix = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign RGB colors to each pixel based on the class labels in the matrix
    for unique_value in np.unique(matrix):
        if unique_value in colordict:
            # Replace the value with the corresponding RGB color
            result_matrix[matrix == unique_value] = colordict[unique_value]
        else:
            # If the value is not in the colordict, keep it unchanged (black)
            result_matrix[matrix == unique_value] = [0, 0, 0]

    return result_matrix
  
def apply_mask_np(image, mask):
  """
  Applies a boolean mask to a RGB PIL image.

  Args:
      image: A 3D NumPy array representing the RGB image.
      mask: A 2D NumPy boolean array representing the mask.

  Returns:
      A masked image (either a subset of the original image or the full image with masked pixels set to zero).
  """
  # Check if the mask has the same spatial dimensions as the image
  if mask.shape[:2] != image.shape[:2]:
    raise ValueError("Mask dimensions must match the image's spatial dimensions.")

  # Expand the mask to 3D if necessary
  if mask.ndim < 3:
    mask_3d = np.repeat(mask[..., None], 3, axis=2)
  else:
    mask_3d = mask

  # Apply the mask using element-wise multiplication
  masked_image = image * mask_3d

  # Alternatively, use boolean indexing for potentially better memory efficiency
  # masked_image = image[mask]

  return masked_image

def split_matrix_into_regions(data,threshold=2):
  """
  Splits a Boolean matrix into a list of contiguous True regions. Handles cases
  with only False values efficiently, ensures returned regions are 2D, and removes
  padding.

  Args:
      data (np.ndarray): A 2D Boolean matrix.

  Returns:
      list: A list of ndarrays, where each element represents a contiguous True region
          in the original unpadded matrix. If the entire matrix is False, an empty list 
          is returned.

  """
  # Check for all False before processing
  if not np.any(data):
    return []
  
  data_total_pixels = data.shape[0] * data.shape[1]

  padded_data = np.pad(data, pad_width=1, mode='constant', constant_values=False)
  labeled_data, num_features = ndimage.label(padded_data)

  # Get original data shape for unpadding
  original_shape = data.shape

  regions = []
  for label in range(1, labeled_data.max() + 1):
    region_mask = labeled_data == label
    # Extract region data and remove padding based on original shape
    pix_area = sum(sum(region_mask))
    
    pix_area_perc = (pix_area / data_total_pixels) * 100
    
    logging.info(f'{data_total_pixels}, {pix_area}, {pix_area_perc}')
    
    if pix_area_perc > threshold:
      regions.append(region_mask[1:-1, 1:-1])
    
  return regions
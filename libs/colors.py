
from libs.lib import *
import numpy as np
from cmocean.cm import phase

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
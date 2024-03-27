import sys
sys.path.append('.')
from libs.lib import *
from configs.options import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


import torch

# creating the paths if they don't exist
create_folderlist([FINETUNING_ROOTPATH])

# TODO: check if the dataset exists, otherwise clone the repository

def get_available_dataset_classes():
    return os.listdir(SURFACE_SAMPLES_ROOTPATH)

def get_class_amounts():
    ret_dict = {}

    for classname in get_available_dataset_classes():
        class_dirpath = os.path.join(SURFACE_SAMPLES_ROOTPATH, classname)

        ret_dict[classname] = len(os.listdir(class_dirpath))

    return ret_dict


def simple_class_listing(zipped=True,shuffle=True):
    surface_classes = []
    image_paths = []

    for classname in get_available_dataset_classes():
        class_dirpath = os.path.join(SURFACE_SAMPLES_ROOTPATH, classname)

        class_images = os.listdir(class_dirpath)

        if shuffle:
            random.shuffle(class_images)

        for sample in class_images:
            image_path = os.path.join(class_dirpath, sample)

            surface_classes.append(classname)
            image_paths.append(image_path)

    if zipped:
        return list(zip(surface_classes, image_paths,strict=True))
    
    else:
        return image_paths, surface_classes
    
def split_by_percentage(data, perc):
  """
  Splits a list into two sublists based on a single percentage, ensuring all elements are included.

  Args:
      data: The list to be split.
      perc: The percentage to include in the first sublist (between 0 and 100).

  Returns:
      A tuple containing two sublists. The first sublist contains 'perc'% of the data (rounded up), 
      and the second sublist contains the remaining data.
  """

  if not (0 <= perc <= 100):
    raise ValueError("Percentage must be between 0 and 100")

  list_len = len(data)
  split_index = int(max(1, list_len * (perc / 100)))  # Ensure at least 1 element in first sublist
  return data[:split_index], data[split_index:]

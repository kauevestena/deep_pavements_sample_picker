import sys,json
sys.path.append('.')
from libs.lib import *
from configs.options import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel

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

def dump_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    
def splitted_class_listing(perc_train=60,outpath=None):
    surface_classes_train = []
    image_paths_train = []

    surface_classes_test = []
    image_paths_test = []

    for classname in get_available_dataset_classes():
        class_dirpath = os.path.join(SURFACE_SAMPLES_ROOTPATH, classname)

        class_images = os.listdir(class_dirpath)

        random.shuffle(class_images)

        train, test = split_by_percentage(class_images, perc_train)

        for sample in train:
            image_path = os.path.join(class_dirpath, sample)

            surface_classes_train.append(classname)
            image_paths_train.append(image_path)

        for sample in test:
            image_path = os.path.join(class_dirpath, sample)

            surface_classes_test.append(classname)
            image_paths_test.append(image_path)

    if isinstance(outpath, str):

        if not outpath.endswith('.json'):
            outpath += '.json'

        as_dict = {
            'train': {
                'labels': surface_classes_train,
                'image_paths': image_paths_train
            },
            'test': {
                'labels': surface_classes_test,
                'image_paths': image_paths_test
            }
        }

        dump_json(as_dict, outpath)

    assert len(image_paths_train) == len(surface_classes_train)
    assert len(image_paths_test) == len(surface_classes_test)

    return image_paths_train, surface_classes_train, image_paths_test, surface_classes_test
    
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


# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


# Define a custom dataset
class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title
import sys
sys.path.append('.')
from libs.lib import *
from configs.options import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


import torch

def get_available_dataset_classes():
    return os.listdir(SURFACE_SAMPLES_ROOTPATH)

def get_available_dataset_classes_numbers():
    ret_dict = {}

    for classname in get_available_dataset_classes():
        class_dirpath = os.path.join(SURFACE_SAMPLES_ROOTPATH, classname)

        ret_dict[classname] = len(os.listdir(class_dirpath))

    return ret_dict


def recover_from_batch_list(batch_list):
    ret_dict = {}

    for entry in batch_list:
        filepath = entry[0]
        tokenized = os.path.normpath(filepath).split(os.path.sep)
        category = tokenized[-2]
        sample = tokenized[-1]

        if category not in ret_dict:
            ret_dict[category] = []

        ret_dict[category].append(sample)

    return ret_dict

def recover_samples(full_dataset,train_dataset, test_dataset):
    indices_train = train_dataset.indices
    indices_test = test_dataset.indices

    train_batch = [full_dataset.samples[i] for i in indices_train]
    test_batch = [full_dataset.samples[i] for i in indices_test]

    return {
        'train': recover_from_batch_list(train_batch),
        'test': recover_from_batch_list(test_batch)
    }

def preprocess_image(image_path):
    img = transforms.ToTensor()(transforms.Image.open(image_path))  # Load and convert to tensor
    img = transforms.Resize(224)(img)  # Resize to 224x224 (common CLIP input size)
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)  # Normalize
    return img

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        # Recursively traverse subfolders to find images and labels
        for root, _, files in os.walk(data_dir):
            label = root.split("/")[-1]
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)
        label = self.labels[idx]
        return image, label
    

import sys
sys.path.append('.')
from libs.lib import *
from configs.options import *

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
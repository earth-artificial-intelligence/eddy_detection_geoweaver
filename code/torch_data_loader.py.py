# Pytorch DataLoader

import numpy as np
from dependency import os
from device_config_and_data_loader import *
from convert_to_pytorch_data_loader import *


# link npz files

data_root = os.path.join(os.path.expanduser("~"), "ML_eddie")
train_folder = os.path.join(data_root, "cds_ssh_1998-2018_10day_interval")
val_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")
train_file = os.path.join(train_folder, "subset_pet_masks_with_adt_1998-1999_lat14N-46N_lon166W-134W.npz")
val_file = os.path.join(val_folder, "subset_pet_masks_with_adt_2019_lat14N-46N_lon166W-134W.npz")


# Data Loader
# set binary = false if we want to distinguish between cyclonic and anticyclonic

binary = False
num_classes = 2 if binary else 3
train_loader, _ = get_eddy_dataloader(train_file, binary=binary, batch_size=batch_size)
val_loader, _ = get_eddy_dataloader(
    val_file, binary=binary, batch_size=batch_size, shuffle=False
)


# Class Distribution check
train_masks = train_loader.dataset.masks.copy()
class_frequency = np.bincount(train_masks.flatten())
total_pixels = sum(class_frequency)


print(
    f"Total number of pixels in training set: {total_pixels/1e6:.2f} megapixels"
    f" across {len(train_masks)} SSH maps\\n"
    f"Number of pixels that are not eddies: {class_frequency[0]/1e6:.2f} megapixels "
    f"({class_frequency[0]/total_pixels * 100:.2f}%)\\n"
    f"Number of pixels that are anticyclonic eddies: {class_frequency[1]/1e6:.2f} megapixels "
    f"({class_frequency[1]/total_pixels * 100:.2f}%)\\n"
    f"Number of pixels that are cyclonic eddies: {class_frequency[2]/1e6:.2f} megapixels "
    f"({class_frequency[2]/total_pixels * 100:.2f}%)\\n"
)



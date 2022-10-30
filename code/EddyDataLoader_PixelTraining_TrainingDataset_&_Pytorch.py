from file_paths import *
from declaring_epochs_size import *
from data_utils import get_eddy_dataloader
from eddy_import import *
import numpy as np
import torch
from get_eddy_dataloader import *
from eddynet import EddyNet

# set binary = false if we want to distinguish between cyclonic and anticyclonic
binary = False
num_classes = 2 if binary else 3
train_loader, _ = get_eddy_dataloader(train_file, binary=binary, batch_size=batch_size)
val_loader, _ = get_eddy_dataloader(
    val_file, binary=binary, batch_size=batch_size, shuffle=False
)

#Looking at the distribution of class frequencies to identify class imbalances
train_masks = train_loader.dataset.masks.copy()
class_frequency = np.bincount(train_masks.flatten())
total_pixels = sum(class_frequency)
print(
    f"Total number of pixels in training set: {total_pixels/1e6:.2f} megapixels"
    f" across {len(train_masks)} SSH maps\n"
    f"Number of pixels that are not eddies: {class_frequency[0]/1e6:.2f} megapixels "
    f"({class_frequency[0]/total_pixels * 100:.2f}%)\n"
    f"Number of pixels that are anticyclonic eddies: {class_frequency[1]/1e6:.2f} megapixels "
    f"({class_frequency[1]/total_pixels * 100:.2f}%)\n"
    f"Number of pixels that are cyclonic eddies: {class_frequency[2]/1e6:.2f} megapixels "
    f"({class_frequency[2]/total_pixels * 100:.2f}%)\n"
)

#Using plot_sample to visualize the dataset we just loaded.
train_loader.dataset.plot_sample(N=3)
plt.savefig("/Users/lakshmichetana/ML_eddies_Output/datasetPlots",bbox="tight")

#Segmentation Model:
num_classes = 2 if binary else 3
model_name = "eddynet"  # we'll log this in Tensorboard
model = EddyNet(num_classes, num_filters=16, kernel_size=3)
if torch.cuda.is_available(): 
    model.to(device="cuda")

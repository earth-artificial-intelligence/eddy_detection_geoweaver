#Looking at the distribution of class frequencies to identify class imbalances

from eddy_import import *
from get_eddy_dataloader import *
import numpy as np
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

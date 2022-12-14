from file_paths import *
from declaring_epochs_size import *
from data_utils import get_eddy_dataloader

# set binary = false if we want to distinguish between cyclonic and anticyclonic
binary = False
num_classes = 2 if binary else 3
train_loader, _ = get_eddy_dataloader(train_file, binary=binary, batch_size=batch_size)
val_loader, _ = get_eddy_dataloader(
    val_file, binary=binary, batch_size=batch_size, shuffle=False
)

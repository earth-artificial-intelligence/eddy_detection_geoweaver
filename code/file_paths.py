#setting the NPZ file paths
from eddy_import import *

data_root = os.path.join(os.path.expanduser("~"), "ML_eddies")
train_folder = os.path.join(data_root, "cds_ssh_1998-2018_10day_interval")
val_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")
train_file = os.path.join(train_folder, "subset_pet_masks_with_adt_1998-2018_lat14N-46N_lon166W-134W.npz")
val_file = os.path.join(val_folder, "subset_pet_masks_with_adt_2019_lat14N-46N_lon166W-134W.npz")

# seting paths and classes for testfiles

from link_npz_files import *

data_root = os.path.join(os.path.expanduser("~"), "ML_eddies")

val_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")

val_file = os.path.join(val_folder, "subset_pet_masks_with_adt_2019_lat14N-46N_lon166W-134W.npz")

binary = False
num_classes = 2 if binary else 3



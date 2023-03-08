# Use the segmask_and_ssh_utils to generate compress file

from data_loader import *
from generate_segmentation_in_parallel import *
from segmask_and_ssh_utils import *

lon_range = (-166, -134)
lat_range = (14, 46)

train_subset = subset_arrays(
    train_masks,
    train_adt,
    train_adt_filtered,
    train_dates,
    lon_range,
    lat_range,
    plot=False,
    resolution_deg=0.25,
    save_folder=train_folder,
)

test_subset = subset_arrays(
    test_masks,
    test_adt,
    test_adt_filtered,
    test_dates,
    lon_range,
    lat_range,
    plot=True,
    resolution_deg=0.25,
    save_folder=test_folder,
)

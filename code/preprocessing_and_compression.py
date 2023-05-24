#  Generate compress segmentaiton mask

import logging

from compression_and_segmentation_utils import *
from dependency import *


data_root = os.path.join(os.path.expanduser("~"), root_path)
train_folder = os.path.join(data_root, "cds_ssh_1998-2018_10day_interval")
test_folder = os.path.join(data_root, "cds_ssh_2019_10day_interval")

example_file = os.path.join(test_folder, "dt_global_twosat_phy_l4_20190110_vDT2021.nc")

# Generate segmentaion mask

logging.getLogger("pet").setLevel(logging.ERROR)

# enter the AVISO filename pattern
# year, month, and day in file_pattern will be filled in get_dates_and_files:
file_pattern = "dt_global_twosat_phy_l4_{year:04d}{month:02d}{day:02d}_vDT2021.nc"

# training set: 1998 - 2018
train_dates, train_files = get_dates_and_files(
    range(1998, 2019 ), range(1, 13), [10], train_folder, file_pattern
)
train_adt, train_adt_filtered, train_masks = generate_masks_in_parallel(
    train_files, train_dates
)


# test set: 2019
test_dates, test_files = get_dates_and_files(
    [2019], range(1, 13), [1,10,20,30], test_folder, file_pattern
)
test_adt, test_adt_filtered, test_masks = generate_masks_in_parallel(
    test_files, test_dates
)

# copress Segmentaion masks to npz files
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

# compress segmask
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

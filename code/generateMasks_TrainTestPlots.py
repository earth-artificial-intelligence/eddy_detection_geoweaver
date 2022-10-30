#getting the test dates and files of training sets from 1998 - 2018 and from training set 2019 and also setting the logging level as ERROR
from eddy_import import *
from importing_multiprocessor import *
from eddy_paths import *
from eddy_plots import *
import logging
from subset_arrays import *
#from Generate_Masks import *
# northern pacific (32x32 degree -> 128x128 pixels)

logging.getLogger("pet").setLevel(logging.ERROR)

# enter the AVISO filename pattern
# year, month, and day in file_pattern will be filled in get_dates_and_files:
file_pattern = "dt_global_twosat_phy_l4_{year:04d}{month:02d}{day:02d}_vDT2021.nc"

# training set: 1998 - 2018
train_dates, train_files = get_dates_and_files(
    range(1998, 2019), range(1, 13), [1, 10, 20, 30], train_folder, file_pattern
)
train_adt, train_adt_filtered, train_masks = generate_masks_in_parallel(
    train_files, train_dates
)


# test set: 2019
test_dates, test_files = get_dates_and_files(
    [2019], range(1, 13), [1, 10, 20, 30], test_folder, file_pattern
)
test_adt, test_adt_filtered, test_masks = generate_masks_in_parallel(
    test_files, test_dates
)


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

plt.savefig('/Users/lakshmichetana/ML_eddies_Output/Train_Test_Subset_Img.png', bbox_inches ="tight")


#getting the test dates and files of training sets from 1998 - 2018 and from training set 2019 and also setting the logging level as ERROR
from eddy_import import *
from importing_multiprocessor import *
from eddy_paths import *
from eddy_plots import *
import logging
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

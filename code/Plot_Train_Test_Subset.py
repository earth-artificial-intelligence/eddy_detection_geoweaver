from subset_arrays import *
from Generate_Masks import *
# northern pacific (32x32 degree -> 128x128 pixels)
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

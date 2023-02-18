from dependency import *
from unzip_utils import *
from get_data import *


os.chdir('/home/chetana/')
current_working_dir = os.getcwd()
print(current_working_dir)

# Directory names
root_dir_name = "ML_eddies"
train_dir_name = "cds_ssh_1998-2018_10day_interval"
test_dir_name = "cds_ssh_2019_10day_interval"

# Build dir paths
root_path = os.path.join(current_working_dir, root_dir_name)
train_path = os.path.join(root_path, train_dir_name)
test_path= os.path.join(root_path, test_dir_name)

# Check if dir exists
is_root_dir_exists = os.path.exists(root_path)
is_train_dir_exists = os.path.exists(train_path)
is_test_dir_exists = os.path.exists(test_path)


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        logger.info("Successfully created folder")
    except:
        logger.error("Something went wrong while creating folder")



if is_root_dir_exists != True:
    print(root_path)
    create_directory(root_path)
    print("created:",root_path)
    create_directory(train_path)
    create_directory(test_path)
    train_file, test_file = download_data()

    unzip_file( os.path.join(current_working_dir,train_file), train_path)
    unzip_file( os.path.join(current_working_dir,test_file), test_path)


if is_root_dir_exists and is_train_dir_exists != True:
    create_directory("cds_ssh_1998-2018_10day_interval")
    train_file = download_train_data()
    unzip_file( os.path.join(current_working_dir,train_file), train_path)

if  is_root_dir_exists and is_test_dir_exists != True:
    create_directory("cds_ssh_2019_10day_interval")
    test_file = download_test_data()
    unzip_file( os.path.join(current_working_dir,test_file), test_path)



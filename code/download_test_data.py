# Scripts download and unzip test files

# Scripts download and unzip test files

import datetime as datetime
import os
from fetch_data_utils import *


prev_date, prev_month, prev_year = get_dates_with_delta(331)

os.chdir(os.path.expanduser("~"))
current_working_dir = os.getcwd()
root_dir_name = "ML_test"
test_data_store = "cds_ssh_test_everyday_interval"

root_path = os.path.join(current_working_dir, root_dir_name)
test_path = os.path.join(root_path, test_data_store)

create_directory(root_path)
create_directory(test_path)

test_zip_file = download_test_date(prev_year, prev_month, prev_date)
unzip_file(os.path.join(current_working_dir, test_zip_file), test_path)





# Scripts download and unzip test files

# Scripts download and unzip test files

import datetime as datetime
import os
from get_data import *
from unzip_utils import *

today = datetime.date.today()
prev_dt_object = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=331)
date = prev_dt_object.date()

prev_date = str(date.day)
prev_month = str(date.month)
prev_year = str(date.year)

os.chdir(os.path.expanduser("~"))
current_working_dir = os.getcwd()
root_dir_name = "ML_eddies"
test_data_store = "cds_ssh_test_everyday_interval"

root_path = os.path.join(current_working_dir, root_dir_name)
test_path = os.path.join(root_path, test_data_store)

test_zip_file = download_test_date(prev_year, prev_month, prev_date)
unzip_file(os.path.join(current_working_dir, test_zip_file), test_path)





# Contains all the utils functions to download and uzip data

import cdsapi
import datetime as datetime

from dependency import logger
from zipfile import ZipFile

client = cdsapi.Client()


def download_train_data():
    try:
        client.retrieve(
            'satellite-sea-level-global',
            {
                'version': 'vDT2021',
                'variable': 'all',
                'format': 'zip',
                'year': [
                    '1998', '1999', '2000',
                    # '2001', '2002', '2003',
                    # '2004', '2005', '2006',
                    # '2007', '2008', '2009',
                    # '2010', '2011', '2012',
                    # '2013', '2014', '2015',
                    # '2016', '2017', '2018',
                ],
                'month': [
                    '01', 
                    # '02', '03',
                    # '04', '05', '06',
                    # '07', '08', '09',
                    # '10', '11', '12',
                ],
                # 'day': ['01', '10', '20', '30'],
                'day': ['01'],
            },
            'train_data.zip')
        return 'train_data.zip'
    except:
        logger.error("Something went wrong while downloading training data")


def download_test_data():
    try:
        client.retrieve(
            'satellite-sea-level-global',
            {
                'version': 'vDT2021',
                'variable': 'daily',
                'format': 'zip',
                'year': ['2019'],
                'month': [
                    '01', '02', '03',
                    # '04', '05', '06',
                    # '07', '08', '09',
                    # '10', '11', '12',
                ],
                # 'day': ['01', '10', '20', '30'],
                'day': ['01'],
            },
            'test_data.zip')
        return 'test_data.zip'
    except:
        logger.error("Something went wrong while downloading test data")


def download_data():
    train_zip_file = download_train_data()
    test_zip_file = download_test_data()
    return train_zip_file, test_zip_file



def download_test_date(year, month, day):
    if len(month) < 2:
        month = '0'+month

    if len(day) < 2:
        day = '0'+day

    fileName = year + "_" + month + "_" + day + "_test.zip"

    try:
        client.retrieve(
            'satellite-sea-level-global',
            {
                'version': 'vDT2021',
                'variable': 'daily',
                'format': 'zip',
                'year': [str(year)],
                'month': [str(month)],
                'day': [str(day)],
            },
            fileName)
        return fileName
    except:
        logger.error("Something went wrong while downloading daily test data")

# unzip data

def unzip_file(zip_file_path, extract_to_path):
    try:
        with ZipFile(zip_file_path) as zip_file_object:          
            zip_file_object.extractall(extract_to_path)
            
    except:
        logger.error("Something went wrong while extracting File" )
        
        
        
def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        logger.info("Successfully created folder")
    except:
        logger.error("Something went wrong while creating folder")     
        
def get_dates_with_delta(delta = 331):
  today = datetime.date.today()
  prev_dt_object = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days= delta)
  date = prev_dt_object.date()
  prev_date = str(date.day)
  prev_month = str(date.month)
  prev_year = str(date.year)
  return prev_date, prev_month, prev_year

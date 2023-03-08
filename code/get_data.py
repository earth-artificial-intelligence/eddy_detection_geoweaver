
from dependency import logger

import cdsapi

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
            		'2001', '2002', '2003',
           			'2004', '2005', '2006',
                    '2007', '2008', '2009',
                    '2010', '2011', '2012',
                    '2013', '2014', '2015',
                    '2016', '2017', '2018',
                ],
                'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
                'day': ['01','10','20','30'],
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
                'variable': 'all',
                'format': 'zip',
                'year': [ '2019' ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': ['01','10','20','30'],
            },
            'test_data.zip')
        return 'test_data.zip'
    except:
        logger.error("Something went wrong while downloading test data")


def download_data():
    train_zip_file = download_train_data()
    test_zip_file = download_test_data()
    return train_zip_file, test_zip_file

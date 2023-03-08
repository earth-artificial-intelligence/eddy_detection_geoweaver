from dependency import logger
from zipfile import ZipFile

def unzip_file(zip_file_path, extract_to_path):
    try:
        with ZipFile(zip_file_path) as zip_file_object:          
            zip_file_object.extractall(extract_to_path)
            
    except:
        logger.error("Something went wrong while extracting File" )


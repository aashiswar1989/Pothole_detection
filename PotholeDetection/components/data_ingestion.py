from pathlib import Path
from PotholeDetection.config_manager.component_config import DataIngestionConfig
from PotholeDetection.logging.logger import logger

import boto3
from botocore.exceptions import NoCredentialsError


class DataIngestion:
    def __init__(self, ingestion_config: DataIngestionConfig):
        self.config = ingestion_config
        self.s3 = boto3.client('s3') 


    def navigate_s3_bucket(self):
        """
        Navigate through S3 bucket and download dataset
        """
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket = self.config.s3_bucket, Prefix = self.config.s3_prefix)

            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('/'):
                        continue

                    root_dir = Path.cwd().parent.parent
                    data_dir_path = root_dir/Path(key).parent
                    data_file_path = root_dir/Path(key)
                    if not data_dir_path.exists():
                        data_dir_path.mkdir(parents=True, exist_ok=True)

                    self.s3.download_file(self.config.s3_bucket, key, data_file_path)
            
            logger.info("Dataset download from S3 bucket completed successfully")

        except Exception as e:
            logger.error("")
            raise e
    
    def initiate_data_ingestion(self):
        """
        Initiate data ingestion from S3 bucket
        """
        try:
            logger.info("Data ingestion started")
            self.navigate_s3_bucket()
            logger.info("Data ingestion completed successfully")

        except Exception as e:
            logger.error("Error in data ingestion")
            raise e
        
if __name__ == "__main__":
    config = DataIngestionConfig()
    obj = DataIngestion(config)
    obj.initiate_data_ingestion()
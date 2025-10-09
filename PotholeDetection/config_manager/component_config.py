from pathlib import Path
from dataclasses import dataclass
from PotholeDetection.constants.constants import *

@dataclass
class DataIngestionConfig():
    dataset: Path = DATASET_PATH
    train_data: Path = TRAIN_DATA
    test_data: Path = TEST_DATA
    val_data: Path = VAL_DATA
    data_yaml_file: str = DATA_YAML
    s3_bucket: str = S3_Bucket
    s3_prefix: str = S3_Prefix


@dataclass
class DataTransformationConfig():
    pass

from pathlib import Path
from dataclasses import dataclass
from PotholeDetection.constants.constants import *

@dataclass
class DataIngestionConfig():
    s3_bucket: str = S3_Bucket
    s3_prefix: str = S3_Prefix
    root_dir: Path = PROJECT_ROOT_DIR
    artifacts_dir: Path = ARTIFACTS_ROOT/'data_ingestion'


@dataclass
class DataIngestionArtifact():
    dataset: Path


@dataclass
class DataTransformationConfig():
    dataset: Path = DATASET_PATH
    train_data: Path = TRAIN_DATA
    test_data: Path = TEST_DATA
    val_data: Path = VAL_DATA
    data_yaml_file: str = DATA_YAML

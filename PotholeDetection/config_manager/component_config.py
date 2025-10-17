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
class DataValidationConfig():
    dataset: Path
    data_split: list = DATA_SPLIT
    supported_img_ext: list = VALID_IMG_EXT
    artifacts_dir: Path = ARTIFACTS_ROOT/'data_validation'

@dataclass
class DataValidationArtifact():
    validation_report: Path
    validation_status: bool
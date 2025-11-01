from pathlib import Path
from dataclasses import dataclass, field

from PotholeDetection.constants.constants import *

@dataclass
class DataIngestionConfig:
    s3_bucket: str = S3_Bucket
    s3_prefix: str = S3_Prefix
    root_dir: Path = PROJECT_ROOT_DIR
    artifacts_dir: Path = INGESTION_ARTIFACTS


@dataclass
class DataIngestionArtifact:
    dataset: Path

@dataclass
class DataValidationConfig:
    dataset: Path
    data_split: list = field(default_factory=lambda: DATA_SPLIT)
    supported_img_ext: list = field(default_factory=lambda: VALID_IMG_EXT)
    artifacts_dir: Path = VALIDATION_ARTIFACTS

@dataclass
class DataValidationArtifact:
    validation_report: Path
    validation_status: bool

@dataclass
class ModelTrainingConfig:
    dataset: Path
    validation_status: bool
    artifacts_dir: Path = TRAINING_ARTIFACTS
    params: dict = field(default_factory=lambda: PARAMS)
    s3_bucket: str = S3_Bucket
    s3_model_key: str = S3_Model_Key


@dataclass
class ModelTrainingArtifact:
    best_model: Path
    last_model: Path
    s3_model_path: str
    s3_uri: str


@dataclass
class ModelEvaluationConfig:
    dataset: Path
    model_path: str = str(TRAINING_ARTIFACTS/'best.pt')
    artifacts_dir: Path = EVALUATION_ARTIFACTS
    model_name: str = PARAMS['model_name']

@dataclass
class ModelEvaluationArtifact:
    eval_report: Path
    eval_results: dict

    
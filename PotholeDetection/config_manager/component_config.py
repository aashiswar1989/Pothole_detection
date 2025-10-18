from pathlib import Path
from dataclasses import dataclass
from PotholeDetection.constants.constants import *

@dataclass
class DataIngestionConfig:
    s3_bucket: str = S3_Bucket
    s3_prefix: str = S3_Prefix
    root_dir: Path = PROJECT_ROOT_DIR
    artifacts_dir: Path = ARTIFACTS_ROOT/'data_ingestion'


@dataclass
class DataIngestionArtifact:
    dataset: Path

@dataclass
class DataValidationConfig:
    dataset: Path
    data_split: list = DATA_SPLIT
    supported_img_ext: list = VALID_IMG_EXT
    artifacts_dir: Path = ARTIFACTS_ROOT/'data_validation'

@dataclass
class DataValidationArtifact:
    validation_report: Path
    validation_status: bool

@dataclass
class ModelTrainingConfig:
    dataset: Path
    validation_status: bool
    artifacts_dir: Path = ARTIFACTS_ROOT/'training'
    model_name: str = MODEL_NAME
    img_size: int = IMG_SIZE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    patience: int = PATIENCE
    optimizer: str = OPTIMIZER
    lr0: float = LR0
    lrf: float = LRF
    momentum: float = MOMENTUM
    weight_decay: float = WEIGHT_DECAY
    workers: int = WORKERS
    warmup_epochs: int = WARMUP_EPOCHS
    val_data: bool = VAL_DATA
    plots: bool = PLOTS
    s3_bucket: str = S3_Bucket
    s3_model_key: str = S3_Model_Key


@dataclass
class ModelTrainingArtifact:
    best_model: Path
    last_model: Path
    s3_model_path: str
    s3_uri: str

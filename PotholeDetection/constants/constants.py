from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT_DIR/'artifacts'
DATASET_PATH = Path(r'C:\AI_ML\Projects\Pothole_Detection\dataset')
TRAIN_DATA = DATASET_PATH/'train'
TEST_DATA = DATASET_PATH/'test'
VAL_DATA = DATASET_PATH/'val'
DATA_YAML = str(DATASET_PATH/'data.yaml')
S3_Bucket = 'pothotle-dataset'
S3_Prefix = 'dataset/'
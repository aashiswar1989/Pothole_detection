from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT_DIR/'artifacts'
S3_Bucket = 'pothotle-dataset'
S3_Prefix = 'dataset/'
S3_Model_Key = 'models/best_model.pt'
DATA_SPLIT = ['train', 'test', 'valid']
VALID_IMG_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
MODEL_NAME = 'yolov8s.pt'
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 5
OPTIMIZER = 'auto'
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WORKERS = 8
WARMUP_EPOCHS = 3
VAL_DATA = True
PLOTS = True


DATASET_PATH = Path(r'C:\AI_ML\Projects\Pothole_Detection\dataset')
TRAIN_DATA = DATASET_PATH/'train'
TEST_DATA = DATASET_PATH/'test'
VAL_DATA = DATASET_PATH/'val'
DATA_YAML = str(DATASET_PATH/'data.yaml')
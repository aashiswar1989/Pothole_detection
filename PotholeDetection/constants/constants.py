from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT_DIR/'artifacts'
INGESTION_ARTIFACTS = ARTIFACTS_ROOT/'data_ingestion'
VALIDATION_ARTIFACTS = ARTIFACTS_ROOT/'data_validation'
TRAINING_ARTIFACTS = ARTIFACTS_ROOT/'training'
EVALUATION_ARTIFACTS = ARTIFACTS_ROOT/'evaluation'
ENV_PATH = PROJECT_ROOT_DIR/'.env'
APP_PATH = PROJECT_ROOT_DIR/'app'
S3_Bucket = 'pothotle-dataset'
S3_Prefix = 'dataset/'
S3_Model_Key = 'models/best_model.pt'
DATA_SPLIT = ['train', 'test', 'valid']
VALID_IMG_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
PARAMS = {
    'model_name' : 'yolov8s.pt',
    'img_size' : 640,
    'epochs': 1,
    'batch_size' : 16,
    'patience' : 5,
    'optimizer' : 'auto',
    'lr0' : 0.01,
    'lrf' : 0.01,
    'momentum' : 0.937,
    'weight_decay' : 0.0005,
    'workers' : 8,
    'warmup_epochs' : 3,
    'val_data' : True,
    'plots' : True
}
METRICS = ['precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95']

# DATASET_PATH = Path(r'C:\AI_ML\Projects\Pothole_Detection\dataset')
# TRAIN_DATA = DATASET_PATH/'train'
# TEST_DATA = DATASET_PATH/'test'
# VAL_DATA = DATASET_PATH/'val'
# DATA_YAML = str(DATASET_PATH/'data.yaml')
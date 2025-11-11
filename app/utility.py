from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv
import boto3

from PotholeDetection.logging.logger import logger
from PotholeDetection.constants.constants import S3_Bucket, S3_Model_Key, ENV_PATH, APP_PATH, ARTIFACTS_ROOT

class ApiUtility:
    def __init__(self):
        logger.info("Loading environment variables for API Utility and initializing S3 client.")
        load_dotenv(ENV_PATH)
        self.s3 = boto3.client('s3')
        MODEL_PATH = APP_PATH/'model_from_s3'
        if not MODEL_PATH.exists():
            MODEL_PATH.mkdir(parents = True, exist_ok = True)

        self.model = MODEL_PATH/'pothole_detection_model.pt'

    def download_model(self):

        if self.model.exists():
            logger.info(f'Model already exists at path: {self.model}. Skipping download from S3 bucket')
            return
        logger.info("Downloading model from S3 bucket")
        self.s3.download_file(Bucket = S3_Bucket,
                              Key = S3_Model_Key,
                              Filename = str(self.model)
                              )
        
        logger.info(f"Model downloaded successfully from S3 bucket and save at path: {self.model}")

    def load_model(self):
        logger.info("Loading the downloaded model for inference")
        model = YOLO(str(self.model))
        logger.info("Model loaded successfully")
        return model
    
    def artifacts_path(self):
        output_path = ARTIFACTS_ROOT/'output'
        if not output_path.exists():
            output_path.mkdir(parents = True, exist_ok = True)

        return output_path
    
    def get_detections(self, results):
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            conf_score = result.boxes.conf.tolist()
            labels = [result.names[cls] for cls in result.boxes.cls.tolist()]

            for box, score, label in zip(boxes, conf_score, labels):
                detections.append({
                    'label': label,
                    'confidence': score,
                    'bounding_box': {
                        'xmin': round(box[0],2),
                        'ymin': round(box[1],2),
                        'xmax': round(box[2],2),
                        'ymax': round(box[3],2)
                        }
                })
        
        return detections
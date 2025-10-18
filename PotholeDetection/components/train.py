from pathlib import Path
from mlflow import runs
import boto3
from ultralytics import YOLO
from shutil import copy2
from PotholeDetection.logging.logger import logger
from PotholeDetection.config_manager.component_config import ModelTrainingConfig, ModelTrainingArtifact


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.s3 = boto3.client("s3")

    def train_model(self):
        """
        Train the YOLO model using the provided configuration.
        Return: Trained model object
        """
        try:
            logger.info("Model training started")
            model = YOLO(self.config.model_name)
            predictions = model.train(
                data = self.config.dataset/'data.yaml',
                imgsz = self.config.img_size,
                epochs = self.config.epochs,
                batch = self.config.batch_size,
                patience = self.config.patience,
                optimizer = self.config.optimizer,
                lr0 = self.config.lr0,
                lrf = self.config.lrf,
                momentum = self.config.momentum,
                weight_decay = self.config.weight_decay,
                workers = self.config.workers,
                warmup_epochs = self.config.warmup_epochs,
                val = self.config.val_data,
                plots = self.config.plots
            )

            runs_folder = Path(predictions.save_dir)
            logger.info(f'Model training completed. Trained model saved at: {runs_folder}')

            return runs_folder
        
        except Exception as e:
            logger.error("Error during model training")
            raise e

    def save_model(self, runs_folder: Path):
        """
        Save the traine model to the artifacts directory
        """
        try:
            logger.info("Saving the trained model")
            
            if not self.config.artifacts_dir.exists():
                self.config.artifacts_dir.mkdir(parents = True, exist_ok = True)

            best_model = runs_folder/'weights'/'best.pt'
            last_model = runs_folder/'weights'/'last.pt'

            copy2(best_model, self.config.artifacts_dir/'best.pt')
            copy2(last_model, self.config.artifacts_dir/'last.pt')

            logger.info(f'Best and Last Models saved at {self.config.artifacts_dir}')

        except Exception as e:
            logger.error("Error in saving the model")
            raise e

    def upload_to_s3(self):
        """
        Upload the trained model to S3 bucket
        """
        try:
            logger.info("Uploading the best mode to S3 bucket for future inference")

            self.s3.upload_file(
                Filename = str(self.config.artifacts_dir/'best.pt'),
                Bucket = self.config.s3_bucket,
                Key = self.config.s3_model_key
                )
            
            s3_uri = f's3://{self.config.s3_bucket}/{self.config.s3_model_key}'
            logger.info(f'Model successfully uploaded to S3 at {s3_uri}')
        
            return s3_uri
        
        except Exception as e:
            logger.error("Error in uploading model to S3")
            raise e

    def initiate_model_training(self) -> ModelTrainingArtifact:
        try:
            logger.info(f'Model trianing started with model: {self.config.model_name}')
            runs_folder = self.train_model()
            self.save_model(runs_folder)
            s3_uri = self.upload_to_s3()

            training_artifacts = ModelTrainingArtifact(
                best_model = self.config.artifacts_dir/'best.pt',
                last_model = self.config.artifacts_dir/'last.pt',
                s3_model_path = self.config.s3_model_key,
                s3_uri = s3_uri
            )
            
            logger.info("Model training finished successfully. Trained models saved and uploaded to S3 bucket.")
            return training_artifacts

        except Exception as e:
            logger.error("Error in model training")
            raise e
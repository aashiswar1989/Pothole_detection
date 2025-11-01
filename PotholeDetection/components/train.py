from pathlib import Path
import json
from dotenv import load_dotenv
from dataclasses import asdict
import mlflow 
import mlflow.pytorch
import boto3
from botocore.exceptions import ClientError
from ultralytics import YOLO
from shutil import copy2
from PotholeDetection.logging.logger import logger
from PotholeDetection.constants.constants import INGESTION_ARTIFACTS, VALIDATION_ARTIFACTS, ENV_PATH, TRAINING_ARTIFACTS
from PotholeDetection.config_manager.component_config import ModelTrainingConfig, ModelTrainingArtifact


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.s3 = boto3.client("s3")

        if not self.config.artifacts_dir.exists():
            self.config.artifacts_dir.mkdir(parents = True, exist_ok = True)


    def train_model(self):
        """
        Train the YOLO model using the provided configuration.
        Return: Trained model object
        """
        try:
            logger.info("Model training started")
            model = YOLO(self.config.params['model_name'])
            predictions = model.train(
                data = str(self.config.dataset/'data.yaml'),
                imgsz = self.config.params['img_size'],
                epochs = self.config.params['epochs'],
                batch = self.config.params['batch_size'],
                patience = self.config.params['patience'],
                optimizer = self.config.params['optimizer'],
                lr0 = self.config.params['lr0'],
                lrf = self.config.params['lrf'],
                momentum = self.config.params['momentum'],
                weight_decay = self.config.params['weight_decay'],
                workers = self.config.params['workers'],
                warmup_epochs = self.config.params['warmup_epochs'],
                val = self.config.params['val_data'],
                plots = self.config.params['plots']
            )

            return predictions
        
        except Exception as e:
            logger.error("Error during model training")
            raise e

    def save_model(self, runs_folder: Path):
        """
        Save the traine model to the artifacts directory
        """
        try:
            logger.info("Saving the trained model")
            
            best_model = runs_folder/'weights'/'best.pt'
            last_model = runs_folder/'weights'/'last.pt'

            copy2(best_model, self.config.artifacts_dir/'best.pt')
            copy2(last_model, self.config.artifacts_dir/'last.pt')

            logger.info(f'Best and Last Models saved at {self.config.artifacts_dir}')

        except Exception as e:
            logger.error("Error in saving the model")
            raise e
        
        
    def s3_file_exist(self, s3_bucket, s3_key):
        """
        Check if the model already exists in S3 bucket
        """
        try:
            self.s3.head_object(Bucket = s3_bucket, Key = s3_key)
            return True
        
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info("Model does not exist in S3 bucket")
                return False
            else:
                raise e


    def upload_to_s3(self):
        """
        Upload the trained model to S3 bucket
        """
        try:
            if not self.s3_file_exist(self.config.s3_bucket, self.config.s3_model_key):
                logger.info("Uploading the best mode to S3 bucket for future inference")

                self.s3.upload_file(
                    Filename = str(self.config.artifacts_dir/'best.pt'),
                    Bucket = self.config.s3_bucket,
                    Key = self.config.s3_model_key
                    )
                logger.info(f'Model successfully uploaded to {self.config.s3_bucket}/{self.config.s3_model_key}')

            else:
                logger.info('Model already exists in S3 bucket. Skipping upload')

            s3_uri = f's3://{self.config.s3_bucket}/{self.config.s3_model_key}'
            
        
            return s3_uri
        
        except Exception as e:
            logger.error("Error in uploading model to S3")
            raise e

    def initiate_model_training(self) -> ModelTrainingArtifact:
        try:
            logger.info("Starting MLFlow for training")
            with mlflow.start_run():
                
                logger.info(f'Logging parameters {self.config.params} to MLFlow')
                mlflow.log_params(self.config.params)

                
                logger.info(f"Model training started with model: {self.config.params['model_name']}")
                predictions = self.train_model()
                runs_folder = Path(predictions.save_dir)
                self.save_model(runs_folder)

                metrics = {
                    'precision': round(predictions.results_dict.get('metrics/precision(B)'), 2),
                    'recall': round(predictions.results_dict.get('metrics/recall(B)'),2),
                    'mAP_0.5': round(predictions.results_dict.get('metrics/mAP50(B)'),2),
                    'mAP_0.5:0.95': round(predictions.results_dict.get('metrics/mAP50-95(B)'),2)
                    }

                logger.info(f'Logging metrics {metrics} to MLFlow')
                mlflow.log_metrics(metrics)  

                confusion_matrix = runs_folder/'confusion_matrix.png'
                pr_curve = runs_folder/'PR_curve.png'
                results_csv = runs_folder/'results.csv'

                logger.info(f'Logging results.csv, confusion_matrix and pr_curve as artifacts to MLFlow')
                mlflow.log_artifact(confusion_matrix, artifact_path = 'confusion_matrix')
                mlflow.log_artifact(pr_curve, artifact_path = 'pr_curve')
                mlflow.log_artifact(results_csv, artifact_path = 'results.csv')


                # Save model to S3                
                s3_uri = self.upload_to_s3()
                mlflow.log_param('s3_uri', s3_uri)

                best_model = self.config.artifacts_dir/'best.pt'
                last_model = self.config.artifacts_dir/'last.pt'
                
                logger.info(f'Logging best_model and last_model as artifacts to MLFlow')
                mlflow.log_artifact(best_model, artifact_path = 'best_model')
                mlflow.log_artifact(last_model, artifact_path = 'last_model')

            
            training_artifacts = ModelTrainingArtifact(
                best_model = best_model,
                last_model = last_model,
                s3_model_path = self.config.s3_model_key,
                s3_uri = s3_uri
            )
            
            training_data = self.config.artifacts_dir/'training_artifacts.json'
            logger.info(f'Creating brief report of model training artifacts at {training_data}')

            with open(training_data, 'w') as f:
                json.dump(asdict(training_artifacts), f, default = str, indent=4)
            
            logger.info("Model training finished successfully. Trained models saved and uploaded to S3 bucket.")
            return training_artifacts

        except Exception as e:
            logger.error("Error in model training")
            raise e
        

if __name__ == "__main__":

    load_dotenv(str(ENV_PATH))

    #load the ingestion artifacts
    if not (INGESTION_ARTIFACTS/'ingestion_artifacts.json').exists():
        logger.error("No artifacts created from Data Ingestion Component")
        raise FileNotFoundError("Ingestion artifact not found. Run data_ingestion stage first.")
    
    with open(INGESTION_ARTIFACTS/'ingestion_artifacts.json', 'r') as f:
        ingestion_artifacts = json.load(f)

    if not (VALIDATION_ARTIFACTS/'validation_report.json').exists():
        logger.error("No artifacts created from Data Validation Component")
        raise FileNotFoundError("Validation artifact not found. Run data_validation stage first.")
    
    with open(VALIDATION_ARTIFACTS/'validation_report.json', 'r') as f:
        validation_artifacts = json.load(f)

    if not validation_artifacts['validation_status']:
        raise Exception("Data Validation Failed. Stopping the pipeline.")

    training_object = ModelTrainingConfig(
        dataset = Path(ingestion_artifacts['dataset']),
        validation_status = validation_artifacts['validation_status']
    )

    model_trainer = ModelTrainer(training_object)
    training_artifacts = model_trainer.initiate_model_training()
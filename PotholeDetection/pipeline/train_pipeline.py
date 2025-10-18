from pathlib import Path
from dotenv import load_dotenv
from PotholeDetection.config_manager.component_config import DataIngestionConfig, DataIngestionArtifact
from PotholeDetection.config_manager.component_config import DataValidationConfig, DataValidationArtifact
from PotholeDetection.config_manager.component_config import ModelTrainingConfig, ModelTrainingArtifact
from PotholeDetection.components.data_ingestion import DataIngestion
from PotholeDetection.components.data_validation import DataValidation
from PotholeDetection.components.train import ModelTrainer

load_dotenv()

ingestion_obj = DataIngestionConfig()
data_ingestion = DataIngestion(ingestion_obj)
ingestion_artifacts = data_ingestion.initiate_data_ingestion()

validation_obj = DataValidationConfig(dataset = ingestion_artifacts.dataset)
data_validation = DataValidation(validation_obj)
validation_artifacts = DataValidation.initiate_data_validation()

if not validation_artifacts.validation_status:
    raise Exception("Data Validation Failed. Stopping the pipeline.")

training_object = ModelTrainingConfig(
    dataset = ingestion_artifacts.dataset,
    validation_status = validation_artifacts.validation_status
)

model_trainer = ModelTrainer(training_object)
training_artifacts = model_trainer.initiate_model_training()
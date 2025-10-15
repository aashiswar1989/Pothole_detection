from pathlib import Path
from PotholeDetection.config_manager.component_config import DataIngestionConfig, DataIngestionArtifact
from PotholeDetection.components.data_ingestion import DataIngestion

ingestion_obj = DataIngestionConfig()
data_ingestion = DataIngestion(ingestion_obj)
ingestion_artifacts = data_ingestion.initiate_data_ingestion()


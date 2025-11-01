import datetime
from pathlib import Path
import json
from ultralytics import YOLO
import mlflow
from PotholeDetection.logging.logger import logger
from PotholeDetection.constants.constants import INGESTION_ARTIFACTS
from PotholeDetection.config_manager.component_config import ModelEvaluationConfig, ModelEvaluationArtifact


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self):
        logger.info('Starting evaluation for trained model')
        
        try:
            with mlflow.start_run():
                logger.info('Logging evaluation parameters to MLFlow')
                mlflow.log_param('model_name', str(self.config.model_name))
                mlflow.log_param('model_path', str(self.config.model_path))

                #Load the model
                model = YOLO(self.config.model_path)
                logger.info(f'Loaded model from {self.config.model_path}')

                test_data_path = self.config.dataset/'data.yaml'

                #evaluate the model
                results = model.val(data = str(test_data_path))
                logger.info("Model evaluation completed successfully")

                metrics = {
                    'precision': round(results.results_dict.get('metrics/precision(B)'), 2),
                    'recall': round(results.results_dict.get('metrics/recall(B)'),2),
                    'mAP_0.5': round(results.results_dict.get('metrics/mAP50(B)'),2),
                    'mAP_0.5:0.95': round(results.results_dict.get('metrics/mAP50-95(B)'),2)
                }

                mlflow.log_metrics(metrics)

                report_content = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Model Name': self.config.model_name,
                    'Trained Model Path': str(self.config.model_path),
                    'Evaluation Metrics': metrics,
                    'confusion matrix': str(results.save_dir/'confusion_matrix.png'),
                    'pr curve': str(results.save_dir/'PR_curve.png')
                }

                if not self.config.artifacts_dir.exists():
                    self.config.artifacts_dir.mkdir(parents = True, exist_ok = True)

                evaluation_report = self.config.artifacts_dir/'evaluation_report.json'
                # metrics_report = self.config.artifacts_dir/'metrics.json'
                logger.info(f'Creating evaluation report at {evaluation_report}')
                with open(evaluation_report, 'w') as f:
                    json.dump(report_content, f, indent = 4)

                logger.info(f'Evaluation report saved at {evaluation_report}')

                mlflow.log_artifact(evaluation_report, artifact_path = 'evaluation_report')


            eval_artifacts = ModelEvaluationArtifact(
                eval_report = evaluation_report,
                eval_results = metrics
            )
            return eval_artifacts

        except Exception as e:
            logger.error('Error during model evaluation')
            raise e
        

if __name__ == "__main__":

    #load the ingestion artifacts
    if not (INGESTION_ARTIFACTS/'ingestion_artifacts.json').exists():
        logger.error("No artifacts created from Data Ingestion Component")
        raise FileNotFoundError("Ingestion artifact not found. Run data_ingestion stage first.")
    
    with open(INGESTION_ARTIFACTS/'ingestion_artifacts.json', 'r') as f:
        ingestion_artifacts = json.load(f)

    eval_object = ModelEvaluationConfig(dataset = Path(ingestion_artifacts['dataset']))
    model_evaluator = ModelEvaluator(eval_object)
    eval_artifacts = model_evaluator.initiate_model_evaluation()
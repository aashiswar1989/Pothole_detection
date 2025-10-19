import datetime
from pathlib import Path
import json
from ultralytics import YOLO
from PotholeDetection.logging.logger import logger
from PotholeDetection.config_manager.component_config import ModelEvaluationConfig, ModelEvaluationArtifact


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self):
        logger.info('Starting evaluation for trained model')
        
        try:
            #Load the model
            model = YOLO(self.config.model_path)
            logger.info(f'Loaded model from {model}')

            test_data_path = self.config.dataset/'data.yaml'

            #evaluate the model
            results = model.val(data = test_data_path)
            logger.info("Model evaluation completed successfully")

            logger.info('Genrating the mdoel evaluation report')

            metrics = {
                'precision': results.results_dict.get('precision'),
                'recall': results.results_dict.get('recall'),
                'mAP_0.5': results.results_dict.get('mAP_0.5'),
                'mAP_0.5:0.95': results.results_dict.get('mAP_0.5:0.95')
            }

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
            with open(evaluation_report, 'w') as f:
                json.dump(report_content, f, indent = 4)

            logger.info(f'Evaluation report saved at {evaluation_report}')

            eval_artifacts = ModelEvaluationArtifact(
                eval_report = evaluation_report,
                eval_results = metrics
            )
            return eval_artifacts

        except Exception as e:
            logger.error('Error during model evaluation')
            raise e
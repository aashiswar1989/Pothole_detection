from pathlib import Path

PROJECT_NAME = "PotholeDetection"

file_list = [
    f'{PROJECT_NAME}/__init__.py',
    f'{PROJECT_NAME}/components/__init__.py',
    f'{PROJECT_NAME}/components/data_ingestion.py',
    f'{PROJECT_NAME}/components/data_transformation.py',
    f'{PROJECT_NAME}/components/train.py',
    f'{PROJECT_NAME}/components/evaluate.py',
    f'{PROJECT_NAME}/utils/__init__.py',
    f'{PROJECT_NAME}/utils/utils.py',
    f'{PROJECT_NAME}/config_manager/__init__.py',
    f'{PROJECT_NAME}/config_manager/component_config.py',
    f'{PROJECT_NAME}/constants/__init__.py',
    f'{PROJECT_NAME}/constants/constants.py',
    f'{PROJECT_NAME}/pipeline/__init__.py',
    f'{PROJECT_NAME}/pipeline/train_pipeline.py',
    f'{PROJECT_NAME}/pipeline/predict_pipeline.py',
    f'{PROJECT_NAME}/logging/__init__.py',
    f'{PROJECT_NAME}/logging/logger.py',
    'experiments/notebook.ipynb',
    'logs/',
    'app/',
    'dvc.yaml',
    '.dockerignore',
    'config.yaml',
    'requirements.txt',
    'requirements_dev.txt',
    'setup.py',
    'params.yaml',
]

for file_path in file_list:
    file, parent = Path(file_path), Path(file_path).parent

    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True)

    if not file.exists():
        file.touch()

# if __name__ == '__main__':
#     project_struct()
#     print(f"Project structure for '{PROJECT_NAME}' created successfully.")
from email.mime import image
from pathlib import Path
from PIL import Image
import json
from ultralytics.models.nas import val
import yaml
from PotholeDetection.config_manager.component_config import DataValidationConfig, DataValidationArtifact
from PotholeDetection.logging.logger import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.invalid_img_ext = []
        self.valid_imgs = 0
        self.total_imgs = 0
        self.total_annotations = 0
        self.missing_annotation_files = []
        self.invalid_annotation_files = []


    def validate_folder_structure(self) -> bool:
        """
        Validate the folder structure of the dataset
        Expected structure:
        dataset/
            - train/
                - images/
                - annotations/
            - test/
                - images/
                - annotations/
            - valid/
                - images/
                - annotations/
        Returns: bool: True if structure is valid, False otherwise

        """
        logger.info("Validating folder structure")
        try:
            for split in self.config.data_split:
                split_path = self.config.dataset/split
                logger.info(f"Checking directory: {split_path}")
                if not split_path.exists():
                    logger.error(f"Missing directory: {split_path}")
                    return False
                
                images_path = split_path/'images'
                annotations_path = split_path/'labels'

                logger.info(f"Checking directory: {images_path}")
                if not images_path.exists():                    
                    logger.error(f"Missing directory: {images_path}")
                    return False
                
                logger.info(f"Checking directory: {annotations_path}")
                if not annotations_path.exists():                    
                    logger.error(f"Missing directory: {annotations_path}")
                    return False
                
            logger.info("Folder structure validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Error during folder structure validation: {e}")
            return False
        
    def validate_image_extensions(self) -> bool:
        """
        Validate image file extensions in the dataset
        Returns: bool: True if all image files have valid extensions, False otherwise
        """
        logger.info("Validating image extensions")
        valid = True
        try:
            for split in self.config.data_split:
                images_path = self.config.dataset/split/'images'

                image_list = [f for f in images_path.rglob('*') if f.is_file()]
                for img_file in image_list:
                    if img_file.suffix.lower() not in self.config.supported_img_ext:
                        self.invalid_img_ext.append(img_file)
                        logger.error(f"Invalid image extension found: {img_file}")
                        valid = False

        except Exception as e:
            logger.error(f"Error during image extension validation: {e}")
            valid = False

        if valid:
            logger.info("Image extension validation passed")
        else:
            logger.error("Image extension validation failed")

        return valid


    def validate_image_files(self) -> bool:
        """
        Validate image files in the dataset
        Checks if images can be opened and are not corrupted
        
        Returns: bool: True if all images are valid, False otherwise
        """
        logger.info("Validating image files")
        valid = True
        try:
            for split in self.config.data_split:
                images_path = self.config.dataset/split/'images'

                image_list = [f for f in images_path.rglob('*') if f.is_file()]
                self.total_imgs = len(image_list)
                for img in image_list:
                    try:
                        img = Image.open(img)
                        img.verify()
                        self.valid_imgs += 1
                    except:
                        logger.error(f"Corrupted image file found: {img}")
                        valid = False
        
        except Exception as e:
            logger.error(f"Error during image file validation: {e}")
            valid = False
        
        if valid:
            logger.info("Image file validation passed")
        else:
            logger.info("Image file validation failed")

        return valid


    def validate_annotations(self) -> bool:
        """
        Validate annotation files in the dataset
        Checks if annotation files are in valid JSON format
        
        Returns: bool: True if all annotations are valid, False otherwise
        """
        logger.info("Validating annotation files")
        valid = True
        try:
            for split in self.config.data_split:
                annotations_path = self.config.dataset/split/'labels'

                annotations_list = [f for f in annotations_path.rglob("*.txt")]
                self.total_annotations = len(annotations_list)
                for ann_file in annotations_list:
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                logger.error(f"Invalid annotation format in file: {ann_file}")
                                self.invalid_annotation_files.append(ann_file)
                                valid = False
                            
                            class_id, x_center, y_center, width, height = parts
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                                logger.error(f"Invalid values for bounding boxes in file: {ann_file}")
                                self.invalid_annotation_files.append(ann_file)
                                valid = False
                            
                            if not class_id.isdigit() or int(class_id) < 0:
                                logger.error(f"Invalid class ID in file: {ann_file}")
                                self.invalid_annotation_files.append(ann_file)
                                valid = False
            
        except Exception as e:
            logger.error(f"Error during annotation file validation: {e}")
            valid = False
        
        if valid: 
            logger.info("Annotation file validation passed")
        else:
            logger.info("Annotation file validation failed")

        return valid


    def validate_image_label_pairs(self) -> bool:
        """
        Validate that each image has a corresponding annotation file
        Returns: bool: True if all images have corresponding annotations, False otherwise
        """
        logger.info("Validating image label pairs")
        valid = True
        try:
            for split in self.config.data_split:
                images_path = self.config.dataset/split/'images'
                annotations_path = self.config.dataset/split/'labels'

                images_list = [f for f in images_path.rglob('*')]
                annotations_list = [f for f in annotations_path.rglob("*.txt")]

                for img_file in images_list:
                    ann_file  = annotations_path/(img_file.stem + '.txt')
                    if not ann_file in annotations_list:
                        logger.error(f"Missing annotation file for image: {img_file}")
                        self.missing_annotation_files.append(img_file)
                        valid = False
        
        except Exception as e:
            logger.error(f"Error during image label pair validation: {e}")
            valid = False   

        if valid:
            logger.info("Image label pair validation passed")
        else:
            logger.info("Image label pair validation failed")

        return valid     

    def validate_yaml_file(self) -> bool:
        """
        Validate the data.yaml file in the dataset
        Checks if the YAML file is well-formed and contains required fields
        Returns: bool: True if YAML file is valid, False otherwise
        """
        logger.info("Validating data.yaml file")
        try:
            yaml_file = self.config.dataset/'data.yaml'
            if not yaml_file.exists():
                logger.error(f"Missing data.yaml file: {yaml_file}")
                return False
            
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            required_fields = ['train', 'val', 'test', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field '{field}' in data.yaml")
                    return False
            
            logger.info("data.yaml file validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Error during data.yaml file validation: {e}")
            return False

    
    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("Data validation started")

        try:
            folder_structure_ok = self.validate_folder_structure()
            image_extensions_ok = self.validate_image_extensions()
            images_ok = self.validate_image_files()
            annotations_ok = self.validate_annotations()
            image_label_mappping_ok = self.validate_image_label_pairs()
            data_yaml_ok = self.validate_yaml_file()

            validation_flag = all([folder_structure_ok, image_extensions_ok, images_ok, annotations_ok, image_label_mappping_ok, data_yaml_ok])

            validation_report = self.config.artifacts_dir/'validation_report.json'
            report_content = {
                "validation_status": validation_flag,
                'folder_structure_ok': folder_structure_ok,
                'image_extensions_ok': image_extensions_ok,
                'images_ok': images_ok,
                'annotations_ok': annotations_ok,
                'image_label_mappping_ok': image_label_mappping_ok,
                'data_yaml_ok': data_yaml_ok,
                'total_images': self.total_imgs,
                'valid_images': self.valid_imgs,
                'total_annotations': self.total_annotations,
                'invalid_image_extensions': self.invalid_img_ext,
                'missing_annotation_files': self.missing_annotation_files,
                'invalid_annotation_files': self.invalid_annotation_files
            }

            if not self.config.artifacts_dir.exists():
                self.config.artifacts_dir.mkdir(exist = True, parents = True)

            with open(validation_report, 'w') as f:
                json.dump(report_content, f, indent = 4)
                logger.info(f'Validation report save to location: {validation_report}')

            validation_artifacts = DataValidationArtifact(
                validation_report = validation_report,
                validation_status = validation_flag
            )
            
            logger.info("Data validation completed")
            return validation_artifacts

        except Exception as e:
            logger.error("Error in data validation")
            raise e
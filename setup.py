from pathlib import Path
from sre_constants import IN
from setuptools import setup, find_packages

PROJECT_NAME = 'PotholeDetection'
AUTHOR = 'Aashish Swarnkar'

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = [line.strip() for line in file.readlines()]

        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name = PROJECT_NAME,
    author = AUTHOR,
    version = '1.0',
    install_requires = get_requirements('requirements_dev.txt'),
    packages = find_packages()
)
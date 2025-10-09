import logging
from pathlib import Path
from datetime import datetime



LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
LOG_PATH = Path('logs')
LOG_FILE_PATH = LOG_PATH / LOG_FILE

if not LOG_PATH.exists():
    LOG_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("Logger has been configured successfully.")
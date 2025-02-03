import logging
from pathlib import Path


def get_model_name() -> str:
    try:
        model_file_path = Path(__file__)
        # Validate the depth of the path
        if len(model_file_path.parts) >= 3:
            model_name = model_file_path.parent.parent.name
            return model_name
        else:
            raise ValueError('Model file path does not match expected structure.')
    except Exception as e:
        logging.error(f'Error obtaining model name: {e}')
        return 'ml_system_logger'


def setup_logger() -> logging.Logger:
    model_name = get_model_name()
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


# Initialize the logger
logger = setup_logger()

import logging
import os

def get_logger(name, log_file='inference.log', log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = logging.FileHandler(f'logs/{log_file}')
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

import logging
import os
from datetime import datetime
import wandb

def get_logger(name: str = "train_logger", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger for training scripts.
    Log file is saved inside the current wandb run directory.

    Args:
        name (str): Name of the logger.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure wandb run is initialized
    if wandb.run is None:
        raise RuntimeError("wandb.run is not initialized. Call wandb.init() first.")

    # Get wandb directory
    wandb_dir = wandb.run.dir

    # Create a unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(wandb_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(level)

    # File handler for logging to wandb directory
    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

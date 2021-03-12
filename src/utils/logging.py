import logging


def get_project_logger(module_name: str) -> logging.Logger:
    return logging.getLogger(f"nlp_gen.{module_name}")

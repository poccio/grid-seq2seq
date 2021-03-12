import subprocess
from typing import Optional, List

import numpy as np

from nlp_gen.utils.logging import get_project_logger

logger = get_project_logger(__name__)


def execute_bash_command(command: str) -> Optional[str]:
    command_result = subprocess.run(command, shell=True, capture_output=True)
    try:
        command_result.check_returncode()
        return command_result.stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning(f"failed executing command: {command}")
        logger.warning(f"return code was: {command_result.returncode}")
        logger.warning(f'stdout was: {command_result.stdout.decode("utf-8")}')
        logger.warning(f'stderr code was: {command_result.stderr.decode("utf-8")}')
        return None


def chunks(l: List, k: int) -> List[List]:
    assert k >= 1
    return [l[i : i + k] for i in range(0, len(l), k)]


def flatten(lst: List[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = np.random.uniform(-noise_value, noise_value)
    return max(1, value + noise)

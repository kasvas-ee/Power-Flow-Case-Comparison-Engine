import yaml
from pathlib import Path
from dataclasses import dataclass, InitVar, fields
from src.logging_config import logger


def check_for_error(config, *keys, expected_type = None, allow_empty=False):
    """
    Fetch the config setting and validate it against the expected datatype
    The setting can be blank only when allow_empty is True
    """
    param_name = ".".join(keys)

    try:
        value = config
        for key in keys:
            value = value[key]

        if value is None:
            if not allow_empty:
                msg = f"Configuration setting {param_name} cannot be empty"
                logger.error(msg)
                raise ValueError(msg)
            else:
                return None

        if expected_type is not None and \
                not isinstance(value, expected_type):
            msg = f"Expected {expected_type.__name__}, got {type(value).__name__} for " \
                  f"configuration setting {param_name}."
            logger.error(msg)
            raise TypeError(msg)

        return value

    except Exception as e:
        msg = f"Configuration setting {param_name} is invalid: {e}"
        logger.error(msg)
        raise RuntimeError(msg)


@dataclass
class Config:
    config_dict: InitVar[dict]
    root_dir: InitVar[Path]

    def __post_init__(self, config_dict: dict, root_dir: Path):
        """
        Transfer the settings from dict into this object
        """
        self.case_a_path = root_dir / check_for_error(config_dict, "Case_inputs", "CaseA_path", expected_type=str)
        self.case_a_format = check_for_error(config_dict, "Case_inputs", "CaseA_format", expected_type=str).lower()
        self.case_b_path = root_dir / check_for_error(config_dict, "Case_inputs", "CaseB_path", expected_type=str)
        self.case_b_format = check_for_error(config_dict, "Case_inputs", "CaseB_format", expected_type=str).lower()

def process_config(config_file: str):
    """
    Read the settings.yaml file
    While reading, validate if the settings provided are of valid datatypes
    """
    try:
        root_dir = Path(__file__).resolve().parents[1]
        config_file = root_dir / config_file
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        config_obj = Config(config_dict=config_dict, root_dir=root_dir)

        return config_obj
    except Exception as e:
        msg = f"Error while loading settings from {config_file}: {e}"
        logger.error(msg)
        raise RuntimeError(msg)
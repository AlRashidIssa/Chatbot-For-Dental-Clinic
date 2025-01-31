from dataclasses import dataclass
import sys, os, yaml
from typing import List

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(MAIN_DIR)


@dataclass
class ConfigPipeline:
    """
    Configuration class to encapsulate all pipeline parameters.
    """
    url: str
    save_archive: str
    name: str
    zip_path: str
    extract_to: str
    database_path: str
    service_columns: List[str]
    branch_columns: List[str]
    social_columns: List[str]
    temperature: float
    do_sample: bool
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    model_id: str
    embedding_model_name: str
    top_k: int
    openai_config_embedding: dict
    openai_embedding: str
    openai_model: str
    openai: bool


def load_config_from_yaml(file_path: str = f"{MAIN_DIR}/config/pipeline_config.yaml") -> ConfigPipeline:
    """
    Load pipeline configuration from a YAML file.

    Parameters:
        file_path (str): Path to the YAML configuration file.

    Returns:
        ConfigPipeline: Configuration object populated with values from the YAML file.
    """
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Create and return a ConfigPipeline object
    return ConfigPipeline(
        url=config_data["url"],
        save_archive=config_data["save_archive"],
        name=config_data["name"],
        zip_path=config_data["zip_path"],
        extract_to=config_data["extract_to"],
        database_path=config_data["database_path"],
        service_columns=config_data["service_columns"],
        branch_columns=config_data["branch_columns"],
        social_columns=config_data["social_columns"],
        temperature=config_data["temperature"],
        do_sample=config_data["do_sample"],
        top_p=config_data["top_p"],
        max_new_tokens=config_data["max_new_tokens"],
        repetition_penalty=config_data["repetition_penalty"],
        model_id=config_data["model_id"],
        embedding_model_name=config_data["embedding_model_name"],
        top_k=config_data["top_k"],
        openai_config_embedding= ,
        openai_embedding= ,
        openai_model= ,
        openai= ,
    )

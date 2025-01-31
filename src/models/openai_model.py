import os
import sys
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import OutputParserException

# Set the main directory path
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import ModelingOperation, HighLevelErrors

# Get OpenAI API Key

class IAPIOpenAIModel(ABC):
    """
    Abstract base class for OpenAI API model loading.

    This class defines the contract for implementing OpenAI model loaders.
    """

    @abstractmethod
    def load(self, model_name: str, config_custom: dict) -> ChatOpenAI:
        """
        Abstract method to load the OpenAI model.

        Args:
            model_name (str): The name of the OpenAI model (e.g., "gpt-4", "gpt-3.5-turbo").
            config_custom (dict): Additional configuration settings for the model.

        Returns:
            ChatOpenAI: An instance of the loaded OpenAI model.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass


class APIOpenAIModel(IAPIOpenAIModel):
    """
    Implementation of the OpenAI API model loader.

    This class provides a method to load an OpenAI Chat model dynamically 
    while handling errors gracefully.
    """

    def load(self, model_name: str, config_custom: dict) -> ChatOpenAI:
        """
        Loads and returns an OpenAI Chat model instance.

        Args:
            model_name (str): The name of the OpenAI model (e.g., "gpt-4", "gpt-3.5-turbo").
            config_custom (dict): Additional configuration settings for the model.

        Returns:
            ChatOpenAI: An instance of the OpenAI chat model.

        Raises:
            ValueError: If the model name is invalid.
            OpenAIError: If there is an issue with OpenAI's API.
            Exception: For any unexpected errors.
        """
        try:
            # Validate model name
            valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            if model_name not in valid_models:
                raise ValueError(
                    f"Invalid model name: {model_name}. Must be one of {valid_models}."
                )
                
            # Load OpenAI model
            return ChatOpenAI(
                model_name=model_name,
                **config_custom
            )

        except ValueError as ve:
            HighLevelErrors.error(f"[Error] ValueError: {ve}")
            raise

        except OutputParserException as ope:
            HighLevelErrors.error(f"[Error] Output Parsing Error: {ope}")
            raise

        except Exception as e:
            HighLevelErrors.error(f"[Error] Unexpected error: {e}")
            raise


# Example usage:
if __name__ == "__main__":
    try:
        model = APIOpenAIModel()
        openai_model = model.load("gpt-4", {"temperature": 0.7})
        ModelingOperation.info("Model loaded successfully: %s", openai_model)
    except Exception as e:
        HighLevelErrors.error(f"Failed to load model: {e}")
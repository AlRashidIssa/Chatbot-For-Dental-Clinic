from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import os, sys

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import ModelingOperation, HighLevelErrors  # Loges

# Define an interface using an Abstract Base Class (ABC)
class EmbeddingModelInterface(ABC):
    @abstractmethod
    def load_model(self, model_name: str):
        pass


# Implementation of the EmbeddingModelInterface
class EmbeddingModelLoader(EmbeddingModelInterface):
    """
    This class implements the EmbeddingModelInterface for loading SentenceTransformer models.
    """
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """Loads the SentenceTransformer model with error handling.
        
        Paramters:
        ----------
        model_name (str): the name Text embedding model (pre-trained)

        Returns:
        --------
        embedding model
        """
        try:
            # Log the attempt to load the model
            ModelingOperation.info(f"Attempting to load model: {model_name}")
            
            # Load the model using SentenceTransformer
            model = SentenceTransformer(model_name)
            
            # Log success after loading the model
            ModelingOperation.info(f"Successfully loaded model: {model_name}")
            
            return model
        
        except ValueError as e:
            # Log and raise ValueError if model loading fails due to a value error
            HighLevelErrors.error(f"Model loading failed due to a ValueError: {e}")
            raise ValueError(f"Model loading failed due to a ValueError: {e}") from e
        
        except OSError as e:
            # Log and raise OSError if model loading fails due to OS-related issues
            HighLevelErrors.error(f"Model loading failed due to an OS error: {e}")
            raise OSError(f"Model loading failed due to an OS error: {e}") from e
        
        except Exception as e:
            # Log and raise any unexpected errors
            HighLevelErrors.error(f"An unexpected error occurred while loading the model: {e}")
            raise RuntimeError(f"An unexpected error occurred while loading the model: {e}") from e
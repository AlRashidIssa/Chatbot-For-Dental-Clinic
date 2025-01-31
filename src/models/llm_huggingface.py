from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from typing import Union, Optional
import os, sys
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import ModelingOperation, HighLevelErrors  # Loges

# Load environment variables from a .env file
load_dotenv(".ven")

# Get Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Log into HuggingFace using the provided access token
login(HUGGINGFACE_TOKEN)

class CustomModelConfig:
    """
    A class to hold custom parameters for the model pipeline.

    Args:
        temperature (float): Sampling temperature.
        do_sample (bool): Whether to sample the output.
        top_p (float): Top-p sampling probability.
        max_new_tokens (int): Maximum number of tokens to generate.
        repetition_penalty (float): Repetition penalty.
    """

    def __init__(self, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.9,
                 max_new_tokens: int = 256, repetition_penalty: float = 1.2):
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    def to_dict(self) -> dict:
        """
        Convert the config to a dictionary.
        """
        return {
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty
        }


class ILoadLLMHuggingFace(ABC):
    """
    Interface for loading HuggingFace models with optional quantization support.
    This interface abstracts the process of loading models from HuggingFace and wrapping them
    into a LangChain pipeline.
    """

    @abstractmethod
    def load(self, model_id: str, quantization: bool,
             quantization_config: Optional[dict] = None, custom_config: Optional[CustomModelConfig] = None) -> HuggingFacePipeline:
        """
        Abstract method to load a HuggingFace model and wrap it in a LangChain pipeline.

        Args:
            model_id (str): The model identifier from HuggingFace.
            quantization (bool): Whether quantization should be applied.
            quantization_config (Optional[dict], optional): Configuration for quantization if enabled.
            custom_config (Optional[CustomModelConfig], optional): Custom model config for pipeline parameters.

        Returns:
            HuggingFacePipeline: The LangChain pipeline object that wraps the model.

        Raises:
            ModelLoadingError: If there is any issue during the model loading process.
        """
        pass


class LoadLLMHuggingFace(ILoadLLMHuggingFace):
    """
    Implementation of ILoadLLMHuggingFace that loads models from HuggingFace with 
    optional quantization and wraps them into a LangChain pipeline.
    This class handles logging, error handling, and monitoring during the loading process.
    """

    def load(self, model_id: str, quantization: bool,
             quantization_config: Optional[dict] = None, custom_config: Optional[CustomModelConfig] = None) -> HuggingFacePipeline:
        """
        Loads the specified HuggingFace model, applies optional quantization, and 
        wraps the model into a LangChain pipeline for text generation.

        Args:
            model_id (str): The HuggingFace model ID to load.
            quantization (bool): Whether quantization is enabled.
            quantization_config (Optional[dict], optional): The configuration for quantization.
            custom_config (Optional[CustomModelConfig], optional): The custom configuration for the pipeline.

        Returns:
            HuggingFacePipeline: A LangChain pipeline with the loaded model.

        Raises:
            ModelLoadingError: If any error occurs during the loading or configuration process.
        """
        try:
            ModelingOperation.info(f"Starting to load the model '{model_id}' from HuggingFace.")
            

            ModelingOperation.info(f"Successfully logged into HuggingFace using the provided token.")
            
            # Handle quantization configuration if enabled
            if quantization:
                if quantization_config is None:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_skip_modules=None
                    )
                ModelingOperation.info(f"Quantization enabled with config: {quantization_config}.")
            else:
                ModelingOperation.info("Quantization is disabled.")

            # Load the tokenizer and model
            ModelingOperation.info(f"Loading tokenizer for model '{model_id}'.")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )

            # Ensure tokenizer padding is set correctly
            tokenizer.pad_token_id = tokenizer.eos_token_id

            # Prepare custom configuration for the pipeline if available
            if custom_config is None:
                custom_config = CustomModelConfig()  # Use default config if not provided

            pipeline_params = custom_config.to_dict()

            # Create a text-generation pipeline with custom parameters
            ModelingOperation.info("Creating text-generation pipeline with custom parameters.")
            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **pipeline_params
            )

            # Wrap the pipeline into LangChain
            ModelingOperation.info(f"Wrapping the pipeline into LangChain.")
            llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

            ModelingOperation.info(f"Model '{model_id}' successfully loaded and wrapped in LangChain pipeline.")
            return llm

        except Exception as e:
            # Capture any error and log it
            HighLevelErrors.error(f"Error occurred during the loading of model '{model_id}': {str(e)}")
            raise HighLevelErrors.error(f"Failed to load model '{model_id}'.", e)


# # Example usage
# if __name__ == "__main__":
#     model_id = "gpt2"  # Example model ID
#     access_token = ""
#     quantization = False
#     quantization_config = None  # Optional: provide a custom configuration for quantization if needed

#     # Custom pipeline parameters
#     custom_config = CustomModelConfig(temperature=0.8, top_p=0.95)

#     try:
#         # Instantiate the loader and load the model with custom config
#         loader = LoadLLMHuggingFace()
#         llm_pipeline = loader.load(model_id, access_token, quantization, quantization_config, custom_config)

#         # You can now use the `llm_pipeline` for inference or further processing
#         ModelingOperation.info("Model is ready for inference.")
    
#     except Exception as e:
#         # Catch and handle errors specific to model loading
#         HighLevelErrors.error(f"An error occurred during model loading: {e}")

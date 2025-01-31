import os
import sys
from abc import ABC, abstractmethod

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import PipelineOperation, HighLevelErrors  # Logging utilities
from data_operation.pull_from_database import PullDataFromDatabaseQuery
from data_operation.combine_dataframe_with_text import CombinedTables
from models.llm_huggingface import LoadLLMHuggingFace, CustomModelConfig
from models.embedding_model import EmbeddingModelLoader
from RetrievalAugmentedGeneration.retrieve_relevant_documents import RetrieveRelevantDocuments
from RetrievalAugmentedGeneration.generative import ChatbotResponse
from configs import load_config_from_yaml, ConfigPipeline

config = load_config_from_yaml()

# Interface for Pipeline
class IFullPipelineChatbot(ABC):
    @abstractmethod
    def run(self, query: str) -> str:
        pass

class FullPipelineChatbot(IFullPipelineChatbot):
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FullPipelineChatbot, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigPipeline):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config
        self._initialized = False
        self.initialize_pipeline()

    def initialize_pipeline(self):
        """Initializes the pipeline only once."""
        if self._initialized:
            PipelineOperation.info("Pipeline already initialized.")
            return

        # Load and process data
        PipelineOperation.info("Pulling data from database.")
        self.service_df, self.branch_df, self.social_df = PullDataFromDatabaseQuery().pull(
            database_path=self.config.database_path
        )
        
        self.service_df_com = CombinedTables().combined(
            columns=self.config.service_columns, df=self.service_df
        )
        self.branch_df_com = CombinedTables().combined(
            columns=self.config.branch_columns, df=self.branch_df
        )
        self.social_df_com = CombinedTables().combined(
            columns=self.config.social_columns, df=self.social_df
        )

        # Load LLM model
        custom_config = CustomModelConfig(
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty
        )
        PipelineOperation.info(f"Loading LLM model: {self.config.model_id}")
        self.llm = LoadLLMHuggingFace().load(
            model_id=self.config.model_id,
            quantization=False,
            quantization_config=None,
            custom_config=custom_config
        )

        # Load embedding model
        PipelineOperation.info(f"Loading embedding model: {self.config.embedding_model_name}.")
        self.embedding_model = EmbeddingModelLoader().load_model(
            model_name=self.config.embedding_model_name
        )

        self._initialized = True
        PipelineOperation.info("Pipeline initialization completed.")

    def run(self, query: str) -> str:
        """Processes a user query and returns the chatbot's response."""
        try:
            relevant_docs = RetrieveRelevantDocuments(top_k=self.config.top_k).relevant(
                query=query,
                embedding_model=self.embedding_model,
                services_df=self.service_df_com,
                branches_df=self.branch_df_com,
                social_media_df=self.social_df_com
            )
            
            response = ChatbotResponse().gen(
                query=query,
                relevant_docs=relevant_docs,
                llm=self.llm
            )
            
            return response  # Return only the assistant's response
        except Exception as e:
            HighLevelErrors.error(f"Error in pipeline: {e}")
            return "An error occurred while processing your request."

if __name__ == "__main__":
    chatbot = FullPipelineChatbot(config=config)
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print(f"Response: {chatbot.run(query)}")

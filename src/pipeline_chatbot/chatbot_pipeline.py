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
from models.openai_model import APIOpenAIModel, APIOpenAIEmbeddingModel
from RetrievalAugmentedGeneration.retrieve_relevant_documents import RetrieveRelevantDocuments
from RetrievalAugmentedGeneration.generative import ChatbotResponse
from configs import load_config_from_yaml, ConfigPipeline

# Load configuration from YAML
config = load_config_from_yaml()

# Interface for the chatbot pipeline
class IFullPipelineChatbot(ABC):
    """Abstract class defining the chatbot pipeline interface."""
    @abstractmethod
    def run(self, query: str) -> str:
        """Runs the chatbot pipeline and returns the response for the given query."""
        pass

class FullPipelineChatbot(IFullPipelineChatbot):
    """A Singleton class implementing the full chatbot pipeline with error handling and logging."""
    
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        """Ensure that only one instance of the chatbot pipeline exists."""
        if cls._instance is None:
            cls._instance = super(FullPipelineChatbot, cls).__new__(cls)
        return cls._instance

    def __init__(self, openai_models: bool, config: ConfigPipeline):
        """Initialize the pipeline only once."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config
        self._initialized = False
        self.openai_models = openai_models
        self.initialize_pipeline()

    def initialize_pipeline(self):
        """Initializes the pipeline and loads necessary models and data."""
        if self._initialized:
            PipelineOperation.info("Pipeline already initialized.")
            return

        try:
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
            if self.openai_models:
                PipelineOperation.info("Loading OpenAI Models [LLM, Embedding]...")
                self.llm = APIOpenAIModel().load(
                    model_name=self.config.openai_model,
                    config_custom=custom_config
                )
                self.embedding_model = APIOpenAIEmbeddingModel().load(
                    model_name=self.config.openai_embedding,
                    config_custom=self.config.openai_config_embedding
                )
            else:
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
        except Exception as e:
            HighLevelErrors.error(f"Error during pipeline initialization: {e}")
            raise

    def run(self, query: str) -> str:
        """Process a user query and return the chatbot's response."""
        try:
            PipelineOperation.info(f"Processing query: {query}")

            # Retrieve relevant documents
            relevant_docs = RetrieveRelevantDocuments(top_k=self.config.top_k).relevant(
                query=query,
                embedding_model=self.embedding_model,
                services_df=self.service_df_com,
                branches_df=self.branch_df_com,
                social_media_df=self.social_df_com
            )
            PipelineOperation.info(f"Retrieved {len(relevant_docs)} relevant documents.")

            # Generate response from the model
            response = ChatbotResponse().gen(
                query=query,
                relevant_docs=relevant_docs,
                llm=self.llm
            )

            PipelineOperation.info(f"Generated response: {response}")
            return response  # Return only the assistant's response
        except Exception as e:
            HighLevelErrors.error(f"Error processing query: {e}")
            return "An error occurred while processing your request."

if __name__ == "__main__":
    """Run the chatbot and interact with the user."""
    chatbot = FullPipelineChatbot(config=config, openai_models=True)

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print(f"Response: {chatbot.run(query)}")

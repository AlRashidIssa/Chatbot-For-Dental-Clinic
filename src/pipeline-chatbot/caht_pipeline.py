import os, sys
from abc import ABC, abstractmethod
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import PipelineOperation, HighLevelErrors  # Loges

from data_operation.ingest_database import LoadFromDrive
from data_operation.unzip_file import UnzipFile
from data_operation.pull_from_database import PullDataFromDatabaseQuery
from data_operation.combine_dataframe_with_text import CombinedTables

from models.llm_huggingface import LoadLLMHuggingFace, CustomModelConfig
from models.embedding_model import EmbeddingModelLoader

from RetrievalAugmentedGeneration.retrieve_relevant_documents import RetrieveRelevantDocuments
from RetrievalAugmentedGeneration.generative import ChatbotResponse

from configs import  load_config_from_yaml, ConfigPipeline

config = load_config_from_yaml()

class IEndToEndPipeline(ABC):
    """
    Abstract base class defining the contract for an end-to-end pipeline.
    Any concrete implementation must define the `run` method.
    """
    @abstractmethod
    def run(self, query: str) -> str:
        """
        Executes the pipeline with a given query.
        
        Parameters:
            query (str): The user query to process.
        
        Returns:
            str: The chatbot's response to the query.
        """
        pass

class EndToEndPipeline(IEndToEndPipeline):
    """
    Concrete implementation of the end-to-end pipeline.
    
    This class initializes and loads necessary data and models only once.
    It processes user queries efficiently by leveraging preloaded data.
    """
    
    def __init__(self, config: ConfigPipeline):
        """
        Initializes the pipeline by loading data, embedding models, and LLM.
        
        Parameters:
            config (ConfigPipeline): The configuration object containing pipeline parameters.
        """
        self.config = config

        # Load data once from the drive
        PipelineOperation.info(f"Starting data ingestion from drive: {self.config.url}")
        LoadFromDrive().load(
            url=self.config.url,
            save_archive=self.config.save_archive,
            name=self.config.name
        )
        
        PipelineOperation.info("Data ingestion completed. Pulling data from database.")
        self.service_df, self.branch_df, self.social_df = PullDataFromDatabaseQuery().pull(
            database_path=self.config.database_path
        )
        
        # Process and structure dataframes
        self.service_df_com = CombinedTables().combined(
            columns=self.config.service_columns,
            df=self.service_df
        )
        self.branch_df_com = CombinedTables().combined(
            columns=self.config.branch_columns,
            df=self.branch_df
        )
        self.social_df_com = CombinedTables().combined(
            columns=self.config.social_columns,
            df=self.social_df
        )

        # Configure the LLM settings
        custom_config = CustomModelConfig(
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty
        )

        # Load LLM model once
        PipelineOperation.info(f"Loading LLM model: {self.config.model_id}")
        self.llm = LoadLLMHuggingFace().load(
            model_id=self.config.model_id,
            access_token=self.config.access_token,
            quantization=False,
            quantization_config=None,
            custom_config=custom_config
        )

        # Load embedding model once
        PipelineOperation.info(f"Loading embedding model: {self.config.embedding_model_name}.")
        self.embedding_model = EmbeddingModelLoader().load_model(
            model_name=self.config.embedding_model_name
        )

    def run(self, query: str) -> str:
        """
        Processes the given query using preloaded data and models to generate a chatbot response.
        
        Parameters:
            query (str): The user query to process.
        
        Returns:
            str: The chatbot's response.
        """
        try:
            # Retrieve relevant documents based on the query
            PipelineOperation.info(f"Retrieving top {self.config.top_k} documents for query: {query}.")
            relevant_docs = RetrieveRelevantDocuments(top_k=self.config.top_k).relevant(
                query=query,
                embedding_model=self.embedding_model,
                services_df=self.service_df_com,
                branches_df=self.branch_df_com,
                social_media_df=self.social_df_com
            )

            # Generate chatbot response
            PipelineOperation.info(f"Generating chatbot response for query: {query}.")
            response = ChatbotResponse().gen(
                query=query,
                relevant_docs=relevant_docs,
                llm=self.llm
            )

            return response

        except Exception as e:
            HighLevelErrors.error(f"Error in pipeline: {e}")
            raise RuntimeError(f"Error running pipeline: {e}")

        
if __name__ == "__main__":
    # Test Case uase
    while True:
        query = input("Enter Question, for quite type exit: ")
        if query == "exit":
            break

        response = EndToEndPipeline().run(query=query,
                                          config=config)
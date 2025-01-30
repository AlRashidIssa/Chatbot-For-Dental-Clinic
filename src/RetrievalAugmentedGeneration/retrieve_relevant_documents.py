import os
import sys
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import ModelingOperation, HighLevelErrors  # Custom logging or monitoring utilities

class IRetrieveRelevantDocuments(ABC):
    """
    Abstract base class for retrieving relevant documents based on embeddings.
    """
    @abstractmethod
    def relevant(self, embedding_model: SentenceTransformer) -> Dict[str, pd.DataFrame]:
        """
        Retrieve relevant documents based on the given embedding model.

        Args:
            embedding_model (SentenceTransformer): The model used to encode query and documents.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing top-k relevant documents for each category.
        """
        pass

class RetrieveRelevantDocuments(IRetrieveRelevantDocuments):
    """
    Implementation of IRetrieveRelevantDocuments for retrieving relevant documents.
    """
    def __init__(self, top_k: int = 10):
        """
        Initialize the retriever.

        Args:
            top_k (int): Number of top relevant documents to retrieve.
        """
        self.top_k = top_k

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build a FAISS index for fast similarity search.

        Args:
            embeddings (np.ndarray): The document embeddings to index.

        Returns:
            faiss.IndexFlatIP: A FAISS index with the embeddings added.
        """
        try:
            ModelingOperation.info("Building FAISS index.")
            index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
            index.add(embeddings)
            ModelingOperation.info("FAISS index built successfully.")
            return index
        except Exception as e:
            ModelingOperation.error(f"Error in building FAISS index: {e}")
            HighLevelErrors.error(f"Error in building FAISS index: {e}")
            raise

    def _retrieve_top_k(self, query_embedding: np.ndarray, index: faiss.IndexFlatIP, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Retrieve top-k documents from the FAISS index.

        Args:
            query_embedding (np.ndarray): The query embedding.
            index (faiss.IndexFlatIP): The FAISS index.
            data (pd.DataFrame): The original DataFrame containing the documents.
            column_name (str): The column name to retrieve results from.

        Returns:
            pd.DataFrame: Top-k most relevant documents.
        """
        try:
            ModelingOperation.info(f"Retrieving top {self.top_k} documents for column {column_name}.")
            faiss.normalize_L2(query_embedding)  # Normalize query embedding
            distances, indices = index.search(query_embedding, self.top_k)

            # Retrieve the rows corresponding to the top indices
            top_k_results = data.iloc[indices[0]]
            top_k_results['similarity'] = distances[0]
            ModelingOperation.info(f"Top {self.top_k} results retrieved for column {column_name}.")
            return top_k_results.sort_values(by='similarity', ascending=False)
        except Exception as e:
            ModelingOperation.error(f"Error in retrieving top-k documents for column {column_name}: {e}")
            HighLevelErrors.error(f"Error in retrieving top-k documents for column {column_name}: {e}")
            raise

    def relevant(self, query: str, embedding_model: SentenceTransformer, services_df: pd.DataFrame, branches_df: pd.DataFrame, social_media_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Retrieve relevant documents across multiple categories.

        Args:
            query (str): The input query string.
            embedding_model (SentenceTransformer): The model used to generate embeddings.
            services_df (pd.DataFrame): DataFrame containing service documents.
            branches_df (pd.DataFrame): DataFrame containing branch documents.
            social_media_df (pd.DataFrame): DataFrame containing social media documents.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing top-k relevant documents for each category.
        """
        try:
            ModelingOperation.info(f"Processing query: {query}")

            # Generate embeddings for the query and documents
            query_embedding = embedding_model.encode([query])
            services_embeddings = embedding_model.encode(services_df['service_name'].tolist())
            branches_embeddings = embedding_model.encode(branches_df['branch_name'].tolist())
            social_media_embeddings = embedding_model.encode(social_media_df['platform_name'].tolist())

            # Build FAISS indices
            services_index = self._build_faiss_index(services_embeddings)
            branches_index = self._build_faiss_index(branches_embeddings)
            social_media_index = self._build_faiss_index(social_media_embeddings)

            # Retrieve top-k results
            top_services = self._retrieve_top_k(query_embedding, services_index, services_df, 'service_name')
            top_branches = self._retrieve_top_k(query_embedding, branches_index, branches_df, 'branch_name')
            top_social_media = self._retrieve_top_k(query_embedding, social_media_index, social_media_df, 'platform_name')

            results = {
                "services": top_services,
                "branches": top_branches,
                "social_media": top_social_media
            }

            ModelingOperation.info(f"Query processed successfully. Returning results.")
            return results
        except Exception as e:
            ModelingOperation.error(f"Error in processing query: {e}")
            HighLevelErrors.error(f"Error in processing query: {e}")
            raise


if __name__ == "__main__":
    try:
        # Example test case
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Example dataframes
        services_df = pd.DataFrame({
            'service_name': ["Account Opening", "Loan Application", "Credit Card Application"]
        })

        branches_df = pd.DataFrame({
            'branch_name': ["Main Branch", "Downtown Branch", "Uptown Branch"]
        })

        social_media_df = pd.DataFrame({
            'platform_name': ["Twitter", "Facebook", "LinkedIn"]
        })

        retriever = RetrieveRelevantDocuments(top_k=2)

        query = "How do I apply for a loan?"
        results = retriever.relevant(query, embedding_model, services_df, branches_df, social_media_df)

        for category, df in results.items():
            print(f"Top results for {category}:")
            print(df)
            print("-")
    except Exception as e:
        ModelingOperation.error(f"Critical failure: {e}")
        HighLevelErrors.error(f"Critical failure: {e}")
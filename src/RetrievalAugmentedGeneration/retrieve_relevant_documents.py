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
    def relevant(self, query: str, embedding_model: SentenceTransformer, services_df: pd.DataFrame, branches_df: pd.DataFrame, social_media_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Retrieve relevant documents based on the given embedding model.
        """
        pass

class RetrieveRelevantDocuments(IRetrieveRelevantDocuments):
    """
    Implementation of IRetrieveRelevantDocuments for retrieving relevant documents.
    """
    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build a FAISS index for fast similarity search.
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings available to build FAISS index.")
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)
        return index

    def _retrieve_top_k(self, query_embedding: np.ndarray, index: faiss.IndexFlatIP, data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Retrieve top-k documents from the FAISS index.
        """
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, min(self.top_k, data.shape[0]))
        
        top_k_results = data.iloc[indices[0]].copy()
        top_k_results.loc[:, 'similarity'] = distances[0]
        return top_k_results.sort_values(by='similarity', ascending=False)

    def relevant(self, query: str, embedding_model: SentenceTransformer, services_df: pd.DataFrame, branches_df: pd.DataFrame, social_media_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Retrieve relevant documents across multiple categories.
        """
        try:
            query_embedding = embedding_model.encode([query])

            results = {}
            for df, col, category in [(services_df, 'service_name', 'services'),
                                       (branches_df, 'branch_name', 'branches'),
                                       (social_media_df, 'platform_name', 'social_media')]:
                if df.empty:
                    results[category] = pd.DataFrame(columns=[col, 'similarity'])
                    continue
                
                embeddings = embedding_model.encode(df[col].tolist())
                index = self._build_faiss_index(embeddings)
                results[category] = self._retrieve_top_k(query_embedding, index, df, col)
            
            return results
        except Exception as e:
            HighLevelErrors.error(f"Error in processing query: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    services_df = pd.DataFrame({'service_name': ["Account Opening", "Loan Application", "Credit Card Application"]})
    branches_df = pd.DataFrame({'branch_name': ["Main Branch", "Downtown Branch", "Uptown Branch"]})
    social_media_df = pd.DataFrame({'platform_name': ["Twitter", "Facebook", "LinkedIn"]})

    retriever = RetrieveRelevantDocuments(top_k=2)
    query = "How do I apply for a loan?"
    results = retriever.relevant(query, embedding_model, services_df, branches_df, social_media_df)
    
    for category, df in results.items():
        print(f"Top results for {category}:")
        print(df)
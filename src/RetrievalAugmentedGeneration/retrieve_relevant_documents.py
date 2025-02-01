import os
import sys
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import faiss

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from utils.monitors import ModelingOperation, HighLevelErrors  # Custom logging or monitoring utilities

# OpenAI API Key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

class IRetrieveRelevantDocuments(ABC):
    """
    Abstract base class for retrieving relevant documents based on embeddings.
    """
    @abstractmethod
    def relevant(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        pass

class RetrieveRelevantDocuments(IRetrieveRelevantDocuments):
    def __init__(self, embedding_model, top_k: int = 10):
        self.top_k = top_k
        self.embedding_model = embedding_model

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build a FAISS index for fast similarity search.
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings available to build FAISS index.")
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)
        return index

    def _retrieve_top_k(self, query_embedding: np.ndarray, index: faiss.IndexFlatIP, data: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve top-k documents from the FAISS index.
        """
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, min(self.top_k, data.shape[0]))
        
        top_k_results = data.iloc[indices[0]].copy()
        top_k_results.loc[:, 'similarity'] = distances[0]

        # Keep only the `combined` column and `similarity`
        return top_k_results[["combined", "similarity"]].sort_values(by="similarity", ascending=False)

    def _encode_with_openai(self, text: str) -> np.ndarray:
        """
        Encode texts using OpenAI embeddings.
        """
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        embedding = np.array(response.data[0].embedding).reshape(1, -1)  # ✅ Correct
        return embedding

    def _encode_with_sentence_transformers(self, texts: list) -> np.ndarray:
        """
        Encode texts using Sentence Transformers.
        """
        return self.embedding_model.encode(texts)

    def relevant(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve relevant documents across all columns combined into one.
        """
        try:
            # Ensure the `combined` column exists
            if "combined" not in df.columns:
                df["combined"] = df.astype(str).agg(' '.join, axis=1)  # Merge all columns into `combined`

            # Choose appropriate encoding method based on the provided embedding model
            if isinstance(self.embedding_model, SentenceTransformer):
                query_embedding = self._encode_with_sentence_transformers([query])
            else:
                query_embedding = self._encode_with_openai(query)

            if query_embedding is None or query_embedding.size == 0:
                raise ValueError("Query embedding is empty. Ensure text is correctly encoded.")

            # Generate embeddings for the `combined` column
            if isinstance(self.embedding_model, SentenceTransformer):
                embeddings = self._encode_with_sentence_transformers(df["combined"].tolist())
            else:
                embeddings = np.vstack([self._encode_with_openai(text) for text in df["combined"].tolist()])
            
            index = self._build_faiss_index(embeddings)
            return self._retrieve_top_k(query_embedding, index, df)
            
        except Exception as e:
            HighLevelErrors.error(f"Error in processing query: {e}")
            return pd.DataFrame(columns=["combined", "similarity"])

if __name__ == "__main__":
    # Example usage:
    
    # Example: Using SentenceTransformer model
    embedding_model_sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Example DataFrame with multiple columns
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "Los Angeles", "Chicago"]
    }
    df = pd.DataFrame(data)

    retriever = RetrieveRelevantDocuments(embedding_model=embedding_model_sentence_transformer, top_k=2)
    query = "Who lives in New York?"
    
    results = retriever.relevant(query, df)

    print("Top relevant results:")
    print(results)

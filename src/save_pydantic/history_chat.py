import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, DateTime, exc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(MAIN_DIR)

# Import chatbot pipeline
from src.utils.monitors import DataOperation, HighLevelErrors

DATABASE_DIR = f"{MAIN_DIR}/Data/SavePydantic"

# Create Dir if it does not exist
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/chat_history.sqlite"

# Define the table structure
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChatHistoryEdge:
    """
    A class to handle the database operations for chat history.
    """
    def __init__(self) -> None:
        """
        Initializes the database connection and creates tables if they do not exist.
        """
        # SQLAlchemy engine and session setup
        self.engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create all tables (this is where the table is created in the database)
        try:
            Base.metadata.create_all(bind=self.engine)
            DataOperation.info("Database and tables created successfully.")
        except exc.SQLAlchemyError as e:
            HighLevelErrors.error(f"Error creating tables: {str(e)}")
            raise

    def get_db(self):
        """Returns a database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def save_chat_response(self, query: str, response: str):
        """
        Saves chat response to the database.

        Args:
            query (str): The user's query.
            response (str): The chatbot's response.

        Returns:
            ChatHistory: The saved chat record.
        """
        db = self.SessionLocal()
        try:
            chat_record = ChatHistory(query=query, response=response)
            db.add(chat_record)
            db.commit()
            db.refresh(chat_record)
            DataOperation.info(f"Chat saved: {chat_record.id}, {chat_record.timestamp}")
            return chat_record
        except exc.SQLAlchemyError as e:
            db.rollback()  # Rollback the transaction if something goes wrong
            HighLevelErrors.error(f"Database error while saving chat: {str(e)}")
            raise 
        except Exception as e:
            db.rollback()
            HighLevelErrors.error(f"Unexpected error while saving chat: {str(e)}")
            raise
        finally:
            db.close()

# Example usage
if __name__ == "__main__":
    chat_edge = ChatHistoryEdge()

    # Example chat interaction
    query = "What is the weather today?"
    response = "The weather is sunny with a high of 75°F."

    # Save the chat interaction
    saved_chat = chat_edge.save_chat_response(query, response)
    print(f"Saved chat record: ID={saved_chat.id}, Query={saved_chat.query}, Response={saved_chat.response}, Timestamp={saved_chat.timestamp}")
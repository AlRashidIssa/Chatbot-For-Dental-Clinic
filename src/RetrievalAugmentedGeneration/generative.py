import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

# Set the main directory path
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Updated LangChaing imports
from langchain.chains.llm import LLMChain
# from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableLambda
from langchain.memory import ConversationBufferMemory

from utils.monitors import ModelingOperation, HighLevelErrors

class IChatbotResponse(ABC):
    """
    Abstract base class for chatbot response generation.
    Subclasses must implement the `gen` method.
    """

    @abstractmethod
    def gen(self, query, relevant_docs: dict, llm):
        """
        Generates a chatbot response based on user input and relevant data.

        Args:
            query (str): User's query.
            relevant_docs (dict): Dictionary containing relevant documents.
            llm (object): Language model used for response generation.

        Returns:
            str: Chatbot's response or an error message.
        """
        pass

class ChatbotResponse(IChatbotResponse):
    """
    Concrete class for generating chatbot responses using an LLM.
    """

    def gen(self, query, relevant_docs: dict, llm):
        """
        Generates a chatbot response based on user input and available data.

        Args:
            query (str): User's query.
            relevant_docs (dict): Dictionary containing relevant data.
            llm (object): Language model used for response generation.

        Returns:
            str: The chatbot's response or an error message.
        """
        try:
            ModelingOperation.info("Starting Generation Response")
            # Step 1: Extract relevant data from documents
            services = ", ".join(relevant_docs.get("services", {}).get("combined", []))
            branches = ", ".join(relevant_docs.get("branches", {}).get("combined", []))
            social_media = ", ".join(relevant_docs.get("social_media", {}).get("combined", []))

            ModelingOperation.info(f"""
                                    The relevant data from documents:
                                    Service: {services}.\n
                                    Branches: {branches}.\n
                                    SOcial_media: {social_media}.\n
                                    """)
            # Define prompt
            prompt = ChatPromptTemplate(
                messages=[
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )

            # Memory for chat history
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Define chain
            chat_chain = prompt | llm

            # Invoke the LLM model
            response = chat_chain.invoke({
                "chat_history": memory.load_memory_variables({}).get("chat_history", []),
                "text": f"""
                You are a helpful chatbot assisting users in Hashmit Kingdom, Jordan.
                ### Information available:
                1. **Available services at our clinic:** {services}
                2. **Our branches in Saudi Arabia:** {branches}
                3. **Social media platforms to contact us:** {social_media}

                ### User's question:
                {query}

                Note: Answer in the same language as the question.
                Note: The Response Markdown Format.
                ### Response:
                """
            })


            # If response is a dictionry, use get.()
            if isinstance(response, dict):
                assistant_message = response.get("text", "").strip().split("Response:")[-1].strip()
            elif hasattr(response, "content"):  # Handle AIMessage case
                assistant_message = response.content.strip().split("Response:")[-1].strip()
            else:
                assistant_message = str(response).strip().split("Response:")[-1].strip()


            ModelingOperation.info("Finishing Gneration Response")
            return assistant_message

        except KeyError as ke:
            error_message = f"Missing required data: {str(ke)}"
            HighLevelErrors.error(f"KeyError: {str(ke)}")
            return error_message
        
        except Exception as e:
            error_message = f"Sorry, an error occurred while processing your request: {str(e)}"
            HighLevelErrors.error(f"Error: {str(e)}")
            return error_message
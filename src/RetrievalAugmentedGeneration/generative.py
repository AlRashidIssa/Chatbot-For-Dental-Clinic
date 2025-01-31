import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

# Updated LangChain imports
from langchain_core.prompts import PromptTemplate  # Corrected import
from langchain.chains import LLMChain  # Corrected import
from langchain.memory import ConversationBufferMemory

from utils.monitors import ModelingOperation, HighLevelErrors  # Assuming these exist

# Set the main directory path
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

class IChatbotResponse(ABC):
    """
    Abstract base class for chatbot response generation.
    Subclasses must implement the `gen` method.
    """

    @abstractmethod
    def gen(self, query, relevant_docs, llm):
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

    def gen(self, query, relevant_docs, llm):
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
            # Step 1: Extract relevant data from documents
            services = ", ".join(relevant_docs.get("services", {}).get("service_name", []))
            branches = ", ".join(relevant_docs.get("branches", {}).get("branch_name", []))
            social_media = ", ".join(relevant_docs.get("social_media", {}).get("platform_name", []))

            # Step 2: Construct input prompt
            input_prompt = f"""
            You are a chatbot operating in Hashmit Kingdom Jordan . Your goal is to assist users by answering their queries about our available services and general inquiries.

            Available services at our clinic:
            {services}

            Our branches in Saudi Arabia:
            {branches}

            You can contact us through the following social media platforms:
            {social_media}

            User question: {query}
            """

            # Step 3: Define the chatbot's prompt template
            template = """<s><|user|>Current conversation:{chat_history}
            {input_prompt}<|end|>
            <|assistant|>"""

            # Step 4: Initialize prompt template
            prompt = PromptTemplate(
                template=template,
                input_variables=["input_prompt", "chat_history"]
            )

            # Step 5: Set up memory to track conversation history
            memory = ConversationBufferMemory(memory_key="chat_history")

            # Step 6: Create LLMChain with prompt and memory
            llm_chain = LLMChain(
                prompt=prompt,
                llm=llm,
                memory=memory
            )

            # Step 7: Invoke the LLM model
            response = llm_chain.invoke({
                "input_prompt": input_prompt,
                "chat_history": memory.load_memory_variables({}).get("chat_history", "")
            })

            # Step 8: Extract the chatbot's response
            assistant_message = response.get("text", "").strip()

            # Step 9: Save conversation context to memory
            memory.save_context(
                inputs={"query": query},
                outputs={"response": assistant_message}
            )

            return assistant_message

        except KeyError as ke:
            error_message = f"Missing required data: {str(ke)}"
            print(f"KeyError: {str(ke)}")
            return error_message
        
        except Exception as e:
            error_message = f"Sorry, an error occurred while processing your request: {str(e)}"
            print(f"Error: {str(e)}")
            return error_message

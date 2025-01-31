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
            ModelingOperation.info("Starating Generation Response")
            # Step 1: Extract relevant data from documents
            services = ", ".join(relevant_docs.get("services", {}).get("service_name", []))
            branches = ", ".join(relevant_docs.get("branches", {}).get("branch_name", []))
            social_media = ", ".join(relevant_docs.get("social_media", {}).get("platform_name", []))

            # Define prompt
            prompt = ChatPromptTemplate(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{text}")
                ]
            )

            # Step 5: Set up memory to track conversation history
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            # chat_history = memory.load_memory_variables({}).get("chat_history", "")

            legacy_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                memory=memory

            )            
            # Step 6: Define a RunnableLambda for model invocation
            # def run_model(inputs):
            #     return legacy_chain.invoke(inputs)

            # chat_chain = prompt | RunnableLambda(run_model)
            

            # Step 7: Invoke the LLM model
            response = legacy_chain.invoke({"text":f"""
                You are a helpful chatbot assisting users in Hashmit Kingdom, Jordan. Your goal is to provide clear, concise, and accurate answers based on the available data. Ensure that your answers are precise, relevant, and easy to understand.

                ### Information available:
                1. **Available services at our clinic:**
                {services}

                2. **Our branches in Saudi Arabia:**
                {branches}

                3. **Social media platforms to contact us:**
                {social_media}

                ### Instructions:
                - Respond to the user's query based on the information provided above.
                - If the query asks about a specific service, branch, or social media platform, provide relevant details.
                - If the query is general or unclear, politely clarify and offer to help further.
                - Be sure to address the user's question completely, providing relevant information when necessary.

                ### User's question:
                {query}
                Note: The answer is in the same language as the question.
                ### Response:
                """})            
            # If response is a dictionry, use get.()
            if isinstance(response, dict):
                # Step 8: Extract the chatbot's response
                assistant_message = response.get("text", "").strip().split("Response:")[-1].strip()
            else:
                # Handle the case when response is a string (or something else)
                assistant_message = response.strip().split("Response:")[-1].strip()

            # # Step 9: Save conversation context to memory
            # memory.save_context(
            #     inputs={"input_prompt": input_prompt},
            #     outputs={"response": assistant_message}
            # )
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
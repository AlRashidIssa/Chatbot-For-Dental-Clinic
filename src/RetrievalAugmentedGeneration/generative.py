import os
import sys
import pandas as pd
from abc import ABC, abstractmethod

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline


from utils.monitors import ModelingOperation, HighLevelErrors  # Assuming these are defined elsewhere

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

class IChatbotResponse(ABC):
    """
    Abstract base class for generating chatbot responses.
    Subclasses must implement the 'gen' method to generate responses.
    """

    @abstractmethod
    def gen(self, query, relevant_docs, llm):
        """
        Generates a chatbot response based on the user's query and available data.
        
        Args:
            query (str): The user's query.
            llm (object): Language model object used to generate the response.

        Returns:
            str: The chatbot's response to the query.
        """
        pass

class ChatbotResponse(IChatbotResponse):
    """
    Concrete class for generating chatbot responses. Implements the 'gen' method.
    """

    def gen(self, query, relevant_docs, llm):
        """
        Generates a response based on the user's query and relevant data from services, branches, and social media.
        
        Args:
            query (str): The user's query.
            relevant_docs
            llm (object): Language model object used to generate the response.

        Returns:
            str: The chatbot's response to the query, or an error message if an error occurs.
        """
        try:

            # Step 2: Extract relevant information from the documents
            services = ", ".join(relevant_docs["services"]["service_name"].tolist())
            branches = ", ".join(relevant_docs["branches"]["branch_name"].tolist())
            social_media = ", ".join(relevant_docs["social_media"]["platform_name"].tolist())

            # Step 3: Construct the chatbot's input prompt with the extracted information
            input_prompt = f"""
            أنت روبوت دردشة في المملكة العربية السعودية. هدفي هو مساعدة المستخدمين في الحصول على إجابات لأسئلتهم حول خدماتنا المتوفرة وأي استفسار عام.

            الخدمات المتوفرة في عيادتنا هي:
            {services}

            الفروع المتوفرة لدينا في المملكة هي:
            {branches}

            يمكنك التواصل معنا عبر منصات الوسائط الاجتماعية التالية:
            {social_media}

            السؤال: {query}
            """

            # Step 4: Prepare the prompt template to include the conversation history
            template = """<s><|user|>Current conversation:{chat_history}
            {input_prompt}<|end|>
            <|assistant|>"""

            # Step 5: Set up the prompt with the appropriate template and input variables
            prompt = PromptTemplate(
                template=template,
                input_variables=["input_prompt", "chat_history"]
            )

            # Step 6: Initialize memory to store and retrieve the conversation history
            memory = ConversationBufferMemory(memory_key="chat_history",)

            # Step 7: Chain the LLM, Prompt, and Memory together
            llm_chain = LLMChain(
                prompt=prompt,
                llm=llm,
                memory=memory
            )

            # Step 8: Invoke the LLM model and capture the response
            response = llm_chain.invoke({
                "input_prompt": input_prompt,
                "chat_history": memory.load_memory_variables({})["chat_history"]
            })

            # Step 9: Extract the assistant's message from the response
            assistant_message = response.get("text", "").strip()

            # Step 10: Save the user query and assistant response to memory
            memory.save_context(
                inputs={"query": query},
                outputs={"response": assistant_message}
            )

            # Return only the assistant's message
            return assistant_message

        except Exception as e:
            # Handle any errors and provide a user-friendly error message
            error_message = f"عذراً، حدث خطأ أثناء معالجة استفسارك: {str(e)}"
            # Optionally, log the error for further inspection
            print(f"Error: {str(e)}")
            return error_message
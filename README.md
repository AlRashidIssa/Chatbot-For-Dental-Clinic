# Chatbot for Dental Clinic

## Overview

This project is a chatbot designed for a dental clinic. It uses machine learning and natural language processing (NLP) techniques to assist patients with appointment scheduling, answering common queries, and providing relevant dental health information.

## Features

- **Conversational AI**: Uses a generative language model for natural interactions.
- **Database Integration**: Retrieves and stores patient information from a database.
- **Document Retrieval**: Fetches relevant documents to provide accurate answers.
- **Monitoring and Logging**: Tracks chatbot performance and logs interactions for analysis.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- pip
- Virtual environment (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Chatbot-For-Dental-Clinic.git
   cd Chatbot-For-Dental-Clinic
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the chatbot by executing:

```bash
python main.py
```

## Project Structure

```
📦src
 ┣ 📂RetrievalAugmentedGeneration
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┣ 📜generative.cpython-312.pyc
 ┃ ┃ ┗ 📜retrieve_relevant_documents.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜generative.py
 ┃ ┗ 📜retrieve_relevant_documents.py
 ┣ 📂__pycache__
 ┃ ┗ 📜__init__.cpython-312.pyc
 ┣ 📂api
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┗ 📜chatbot_api.cpython-312.pyc
 ┃ ┣ 📂static
 ┃ ┃ ┗ 📜styles.css
 ┃ ┣ 📂templates
 ┃ ┃ ┗ 📜chat.html
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜chatbot_api.py
 ┣ 📂cash
 ┃ ┗ 📜__init__.py
 ┣ 📂configs
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜__init__.cpython-312.pyc
 ┃ ┗ 📜__init__.py
 ┣ 📂data_operation
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┣ 📜combine_dataframe_with_text.cpython-312.pyc
 ┃ ┃ ┣ 📜ingest_database.cpython-312.pyc
 ┃ ┃ ┣ 📜pull_from_database.cpython-312.pyc
 ┃ ┃ ┗ 📜unzip_file.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜combine_dataframe_with_text.py
 ┃ ┣ 📜ingest_database.py
 ┃ ┣ 📜pull_from_database.py
 ┃ ┗ 📜unzip_file.py
 ┣ 📂models
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┣ 📜embedding_model.cpython-312.pyc
 ┃ ┃ ┣ 📜llm_huggingface.cpython-312.pyc
 ┃ ┃ ┗ 📜openai_model.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜embedding_model.py
 ┃ ┣ 📜llm_huggingface.py
 ┃ ┗ 📜openai_model.py
 ┣ 📂pipeline_chatbot
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┗ 📜chatbot_pipeline.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜chatbot_pipeline.py
 ┣ 📂save_pydantic
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┗ 📜history_chat.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜history_chat.py
 ┣ 📂utils
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-312.pyc
 ┃ ┃ ┣ 📜get_size.cpython-312.pyc
 ┃ ┃ ┗ 📜monitors.cpython-312.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜get_size.py
 ┃ ┗ 📜monitors.py
 ┗ 📜__init__.py
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, contact alrashidissa2001@hotmail.com or open an issue on GitHub.

![Chat Image](https://example.com/path/to/image.png)

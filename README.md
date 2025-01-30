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
Chatbot-For-Dental-Clinic/
│── src/
│   ├── main.py  # Entry point of the chatbot
│   ├── data_operation/
│   │   ├── ingest_database.py  # Database ingestion logic
│   │   ├── pull_from_database.py  # Fetching data
│   ├── models/
│   │   ├── embedding_model.py  # Text embedding model
│   │   ├── llm_huggingface.py  # LLM integration
│   ├── utils/
│   │   ├── get_size.py  # Utility functions
│   │   ├── monitors.py  # Performance monitoring
│── requirements.txt  # Dependencies
│── README.md  # Documentation
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, contact [alrashidissa2001@hotmail.com] or open an issue on GitHub.



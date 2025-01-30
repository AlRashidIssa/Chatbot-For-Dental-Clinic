from rich.console import Console
from rich.markdown import Markdown

# Initialize the console
console = Console()

# Define markdown content
markdown_content = """
# Full Pipeline for Handling Database from Google Drive to Pass to RAG

## Overview:
This pipeline aims to automate the process of downloading a database from Google Drive, extracting necessary data, and combining it into a structure that is ready for further processing by a RAG (Retrieval-Augmented Generation) system. The flow involves several stages:

1. **Downloading Data from Google Drive**: Using a utility to fetch the database file from a Google Drive URL and save it locally.
2. **Loading the Database**: Once the database file is downloaded, an SQLite database is used to fetch specific tables that contain bilingual data (Arabic and English).
3. **Merging Data**: After loading the data, tables from both languages (Arabic and English) are combined into single DataFrames for each category (services, branches, and social media).
4. **Combining Columns**: Columns from these tables are merged into a single 'combined' column to create a unified dataset that can be directly consumed by downstream processes.
5. **Preparing Data for RAG**: The processed data will then be used as input for the RAG pipeline to retrieve relevant information for generating answers or insights.

## Steps:
### 1. **Download Database from Google Drive**:
   - **Input**: Google Drive URL, Target Directory, File Name.
   - The `LoadFromDrive` class is responsible for downloading the database file (stored as a ZIP) from Google Drive using a valid URL. 
   - It ensures that the target directory exists, extracts the Google Drive file ID, and downloads the file.
   - **Output**: The ZIP file is saved to a designated directory, ready for extraction.
"""

# Create Markdown object
md = Markdown(markdown_content)

# Print it using the console with markdown rendering
console.print(md)
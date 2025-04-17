# Angel One Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit that provides support for Angel One's services by answering questions based on their support documentation and other Insurance documents.

## Features

- Web scraping of Angel One's support documentation
- Document processing and chunking
- Vector-based semantic search using FAISS
- Question answering powered by Google's Gemini model
- User-friendly Streamlit web interface

## Prerequisites

- Python 3.8 or higher
- Google AI API key (for Gemini model)
- Internet connection (for web scraping)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
pip install langchain-huggingface  # Additional package for embeddings
```

4. Set up your environment variables:
```bash
# Create a .env file
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```
Replace `your-api-key-here` with your actual Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Project Structure

```
.
├── app.py              # Streamlit web application
├── ingest.py           # Document processing and ingestion
├── rag_chatbot.py      # RAG implementation with Gemini
├── requirements.txt    # Python dependencies
└── data/              # Directory for storing processed documents
    └── web_pages/     # Scraped web pages
    └── pages/         # Insurance pdfs
```

## Usage

1. First, run the document ingestion process:
```bash
python ingest.py
```
This will:
- Scrape Angel One's support documentation
- loads insurance pdfs
- Process and chunk the documents
- Store them in the data directory

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Enter your questions in the text input field and press Enter to get answers

## How It Works

1. **Document Processing** (`ingest.py`):
   - Scrapes Angel One's support website
   - Processes and chunks the documents
   - Uses BeautifulSoup for web scraping
   - Implements text chunking with LangChain's RecursiveCharacterTextSplitter

2. **RAG Implementation** (`rag_chatbot.py`):
   - Uses HuggingFace embeddings for document vectorization
   - Implements FAISS vector store for efficient similarity search
   - Integrates Google's Gemini model for answer generation
   - Custom LangChain integration for the Gemini model

3. **Web Interface** (`app.py`):
   - Streamlit-based user interface
   - Caches the bot initialization for better performance
   - Provides a simple question-answering interface

## Dependencies

- langchain
- langchain-community
- langchain-core
- langchain-huggingface
- faiss-cpu
- streamlit
- beautifulsoup4
- unstructured
- PyPDF2
- requests
- tqdm
- python-dotenv

## Troubleshooting

1. **API Key Issues**:
   - Ensure your Google API key is correctly set in the .env file
   - Verify the key has access to the Gemini model

2. **Document Processing**:
   - If web scraping fails, check your internet connection
   - Verify the target website's structure hasn't changed

3. **Memory Issues**:
   - If you encounter memory errors, try reducing the chunk size in `ingest.py`
   - Consider using a smaller embedding model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Google AI](https://ai.google.dev/) for the Gemini model
- [Streamlit](https://streamlit.io/) for the web interface 
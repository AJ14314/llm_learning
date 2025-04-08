# LLM Learning Project

A comprehensive project for experimenting with Large Language Models (LLMs), vector databases, and AI tools. This repository includes implementations of RAG, stock analysis, and various LLM integrations.

## 🚀 Features

- **Multiple LLM Integrations**
  - OpenAI
  - Google Gemini
  - Anthropic Claude
  - Ollama for local LLM deployment

- **Vector Databases**
  - FAISS
  - ChromaDB
  - Document embeddings with sentence-transformers

- **Tools & Frameworks**
  - LangChain for orchestration
  - Hugging Face Transformers
  - Gradio for UI interfaces
  - Jupyter for interactive development

- **Data Processing**
  - Stock analysis tools
  - Unstructured data handling
  - Audio processing with pydub
  - Web scraping with BeautifulSoup4

## 📦 Dependencies

Core dependencies include:

```txt
# AI/ML
torch
transformers
openai
google-generativeai
anthropic
ollama
sentence_transformers
bitsandbytes

# Data Processing
numpy
pandas
scipy
scikit-learn

# Vector Databases
faiss-cpu
chromadb

# LangChain Ecosystem
langchain
langchain-openai
langchain_experimental
langchain_chroma

# Visualization
plotly
matplotlib
jupyter-dash

# Development Tools
jupyterlab
python-dotenv
gradio
modal
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm_learning.git
cd llm_learning
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Setup environment variables:
```bash
copy .env.example .env
# Edit .env with your API keys
```

## 📚 Project Structure

```
llm_learning/
├── Practise_Codes/
│   ├── Gemini.ipynb           # Google Gemini experiments
│   ├── HuggingFace.ipynb      # HuggingFace models usage
│   ├── Tokenizers.ipynb       # Tokenization experiments
│   ├── project_s.ipynb        # Stock analysis
│   ├── rag_implementation_chrome.ipynb
│   ├── rag_implementation_faiss.ipynb
│   └── nse_stock_analysis/    # Stock analysis tools
├── requirements.txt           # Python dependencies
└── .env                      # Environment variables
```

## 💡 Usage

1. Start JupyterLab:
```bash
jupyter lab
```

2. Navigate to `Practise_Codes/` and explore the notebooks:
   - `Gemini.ipynb` for Google Gemini experiments
   - `rag_implementation_*.ipynb` for RAG implementations
   - `project_s.ipynb` for stock analysis

## 🔧 Development Tools

- **Code Acceleration**
  - `accelerate` for hardware optimization
  - `bitsandbytes` for quantization
  - `psutil` for system monitoring

- **Data Sources**
  - `feedparser` for RSS feeds
  - `speedtest-cli` for network testing
  - `beautifulsoup4` for web scraping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain community
- Hugging Face team
- ChromaDB developers
- FAISS developers

---
*Note: Make sure to check the individual notebook documentation for specific usage examples and requirements.*
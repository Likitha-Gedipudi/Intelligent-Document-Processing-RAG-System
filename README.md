# 🏦 Banking Document RAG System

A **100% free, locally-running** Retrieval-Augmented Generation (RAG) system for banking documents. Perfect for students and developers who want to learn RAG without paying for expensive APIs.

## ✨ Features

- **📄 Document Processing**: Upload PDFs, TXT, DOCX files
- **🔍 Entity Extraction**: Automatically extract PAN, Aadhaar, account numbers, amounts, dates
- **💬 RAG Chat**: Ask questions about your documents in natural language
- **📊 Dashboard**: View statistics and manage documents
- **🔒 Privacy First**: Everything runs locally - no data leaves your machine

## 🛠️ Tech Stack (All Free!)

| Component | Technology |
|-----------|------------|
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (local) |
| LLM | Ollama (local) |
| Database | SQLite |
| UI | Streamlit |
| PDF Processing | PyPDF2 |

## 🚀 Quick Start

### 1. Clone and Install

```bash
cd "/Users/likig/Desktop/Intelligent Document Processing & RAG System"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama (Optional but Recommended)

For AI-powered answers, install Ollama:

```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai
```

Then pull a model:

```bash
# Standard model (needs ~8GB RAM)
ollama pull llama3:8b

# OR smaller model (needs ~4GB RAM)
ollama pull phi3:mini
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
├── app.py                 # Streamlit main app
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── src/
│   ├── document_processor.py  # PDF extraction & classification
│   ├── entity_extractor.py    # Regex-based NER
│   ├── rag_engine.py          # RAG with embeddings + LLM
│   └── database.py            # SQLite operations
├── data/
│   ├── documents/         # Uploaded files
│   └── chroma_db/         # Vector database
└── sample_docs/           # Test documents
    ├── loan_application.txt
    ├── kyc_document.txt
    └── bank_statement.txt
```

## 💡 Usage

### Upload Documents
1. Use the sidebar to upload PDF, TXT, or DOCX files
2. Click "Process Document"
3. View extracted entities and quality score

### Ask Questions
- "What PAN numbers are in the documents?"
- "Summarize the loan application"
- "What is the account holder's name?"
- "List all transaction amounts"

### Without Ollama
The app works without Ollama installed - you can still:
- Upload and process documents
- Extract entities
- Search documents (returns relevant excerpts)

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

## 📝 License

MIT License - Free for educational and personal use.

---

Built with ❤️ for learning RAG without breaking the bank!

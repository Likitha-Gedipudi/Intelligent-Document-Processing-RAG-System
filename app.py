"""
Banking Document RAG System
A free, locally-running RAG application for banking documents

Author: Student Project
Stack: Streamlit + SQLite + ChromaDB + Sentence Transformers + Ollama
"""
import streamlit as st
import uuid
import os
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS
from src.document_processor import extract_text, classify_document, chunk_text, get_document_stats
from src.entity_extractor import extract_entities, get_entity_summary, calculate_quality_score
from src.database import (
    save_document, save_entities, save_chunks, get_all_documents,
    get_document, get_document_entities, get_statistics, delete_document
)
from src.rag_engine import get_rag_engine

# Page config
st.set_page_config(
    page_title="Banking Doc RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .entity-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px;
        background: #e3f2fd;
        border-radius: 16px;
        font-size: 0.85rem;
    }
    .doc-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def process_uploaded_file(uploaded_file) -> dict:
    """Process an uploaded file through the full pipeline"""
    
    # Generate unique ID
    doc_id = str(uuid.uuid4())
    
    # Save file to disk
    file_path = DOCUMENTS_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text
    text = extract_text(str(file_path))
    
    # Classify document
    doc_type = classify_document(text)
    
    # Extract entities
    entities = extract_entities(text)
    entity_summary = get_entity_summary(entities)
    
    # Calculate quality score
    quality_score = calculate_quality_score(text, doc_type)
    
    # Get document stats
    stats = get_document_stats(text)
    
    # Chunk text for RAG
    chunks = chunk_text(text)
    
    # Save to SQLite
    save_document(
        doc_id=doc_id,
        filename=uploaded_file.name,
        file_path=str(file_path),
        file_size=uploaded_file.size,
        doc_type=doc_type,
        extracted_text=text,
        quality_score=quality_score,
        metadata={**stats, 'entity_count': entity_summary['total_entities']}
    )
    
    # Save entities
    save_entities(doc_id, entities)
    
    # Add to vector database
    rag_engine = get_rag_engine()
    embedding_ids = rag_engine.add_document(
        doc_id=doc_id,
        chunks=chunks,
        metadata={
            'filename': uploaded_file.name,
            'doc_type': doc_type,
            'quality_score': quality_score
        }
    )
    
    # Save chunk references
    save_chunks(doc_id, chunks, embedding_ids)
    
    return {
        'doc_id': doc_id,
        'filename': uploaded_file.name,
        'doc_type': doc_type,
        'entities': entities,
        'entity_summary': entity_summary,
        'quality_score': quality_score,
        'stats': stats,
        'chunks_created': len(chunks)
    }


def render_sidebar():
    """Render the sidebar with upload and navigation"""
    
    with st.sidebar:
        st.markdown("## 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[ext.replace('.', '') for ext in SUPPORTED_EXTENSIONS],
            help="Supported: PDF, TXT, DOCX"
        )
        
        if uploaded_file:
            st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        result = process_uploaded_file(uploaded_file)
                        st.session_state['last_processed'] = result
                        st.success("✅ Document processed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 📑 Navigation")
        page = st.radio(
            "Go to",
            ["💬 Chat & Query", "📚 Documents", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        stats = get_statistics()
        st.markdown("### 📈 Quick Stats")
        st.metric("Documents", stats.get('total_documents', 0))
        st.metric("Entities", stats.get('total_entities', 0))
        st.metric("Avg Quality", f"{stats.get('avg_quality_score', 0):.0f}%")
        
        return page


def render_chat_page():
    """Render the main chat/query interface"""
    
    st.markdown('<p class="main-header">💬 Ask Questions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query your banking documents using natural language</p>', unsafe_allow_html=True)
    
    # Check Ollama status
    rag_engine = get_rag_engine()
    rag_stats = rag_engine.get_collection_stats()
    
    if not rag_stats.get('ollama_available'):
        st.warning("""
        ⚠️ **Ollama not detected.** For full AI-powered answers, install Ollama:
        1. Download from [ollama.ai](https://ollama.ai)
        2. Run: `ollama pull llama3:8b` (or `phi3:mini` for lower RAM)
        3. Restart this app
        
        *You can still upload documents and search - just without AI summarization.*
        """)
    
    # Show last processed document
    if 'last_processed' in st.session_state:
        result = st.session_state['last_processed']
        with st.expander(f"✅ Recently processed: {result['filename']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Type", result['doc_type'].replace('_', ' ').title())
            col2.metric("Quality Score", f"{result['quality_score']:.0f}%")
            col3.metric("Chunks", result['chunks_created'])
            
            st.markdown("**Extracted Entities:**")
            for entity_type, values in result['entities'].items():
                st.markdown(f"- **{entity_type.replace('_', ' ').title()}:** {', '.join(values[:5])}")
    
    # Query input
    st.markdown("### Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="e.g., What PAN numbers are in the documents?",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Filter options
    with st.expander("🔧 Search Options"):
        filter_type = st.selectbox(
            "Filter by document type",
            ["All Types", "Loan Application", "KYC Document", "Bank Statement", "Salary Slip", "Other"]
        )
        top_k = st.slider("Number of results", 1, 10, 5)
    
    # Process query
    if search_btn and query:
        filter_value = None
        if filter_type != "All Types":
            filter_value = filter_type.lower().replace(' ', '_')
        
        with st.spinner("Searching..."):
            result = rag_engine.query(
                question=query,
                top_k=top_k,
                filter_doc_type=filter_value
            )
        
        # Display answer
        st.markdown("### 💡 Answer")
        st.markdown(result['answer'])
        
        # Display sources
        if result['sources']:
            st.markdown("### 📄 Sources")
            for source in result['sources']:
                st.markdown(
                    f"- **{source['filename']}** ({source['doc_type'].replace('_', ' ').title()}) "
                    f"- Relevance: {source['relevance']:.0%}"
                )
        
        # Display timing
        st.caption(
            f"⏱️ Search: {result['search_time_ms']}ms | "
            f"Total: {result['total_time_ms']}ms | "
            f"Chunks retrieved: {result['chunks_retrieved']}"
        )
    
    # Example queries
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "What PAN numbers are in the uploaded documents?",
        "Summarize the loan application details",
        "What is the account holder's name?",
        "List all the dates mentioned in the documents",
        "What is the loan amount requested?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state['example_query'] = example
                st.rerun()


def render_documents_page():
    """Render the documents list page"""
    
    st.markdown('<p class="main-header">📚 Documents</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View and manage uploaded documents</p>', unsafe_allow_html=True)
    
    documents = get_all_documents()
    
    if not documents:
        st.info("No documents uploaded yet. Use the sidebar to upload your first document!")
        return
    
    # Filter
    doc_types = list(set(d['doc_type'] for d in documents))
    filter_type = st.selectbox("Filter by type", ["All"] + doc_types)
    
    filtered_docs = documents
    if filter_type != "All":
        filtered_docs = [d for d in documents if d['doc_type'] == filter_type]
    
    st.markdown(f"Showing {len(filtered_docs)} document(s)")
    
    # Display documents
    for doc in filtered_docs:
        with st.expander(f"📄 {doc['filename']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Type", doc['doc_type'].replace('_', ' ').title())
            col2.metric("Quality", f"{doc['quality_score']:.0f}%")
            col3.metric("Size", f"{doc['file_size'] / 1024:.1f} KB")
            
            st.markdown(f"**Uploaded:** {doc['upload_date'][:19]}")
            
            # Show entities
            entities = get_document_entities(doc['id'])
            if entities:
                st.markdown("**Extracted Entities:**")
                for etype, values in entities.items():
                    st.markdown(f"- {etype.replace('_', ' ').title()}: `{', '.join(values[:3])}`{'...' if len(values) > 3 else ''}")
            
            # Show text preview
            if doc['extracted_text']:
                st.markdown("**Text Preview:**")
                st.text(doc['extracted_text'][:500] + "..." if len(doc['extracted_text']) > 500 else doc['extracted_text'])
            
            # Delete button
            if st.button("🗑️ Delete", key=f"delete_{doc['id']}"):
                rag_engine = get_rag_engine()
                rag_engine.delete_document_embeddings(doc['id'])
                delete_document(doc['id'])
                # Also delete the file
                try:
                    os.remove(doc['file_path'])
                except:
                    pass
                st.success("Document deleted!")
                st.rerun()


def render_dashboard_page():
    """Render the analytics dashboard"""
    
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analytics and system status</p>', unsafe_allow_html=True)
    
    stats = get_statistics()
    rag_engine = get_rag_engine()
    rag_stats = rag_engine.get_collection_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📄 Total Documents", stats.get('total_documents', 0))
    col2.metric("🏷️ Total Entities", stats.get('total_entities', 0))
    col3.metric("💬 Total Queries", stats.get('total_queries', 0))
    col4.metric("📊 Avg Quality", f"{stats.get('avg_quality_score', 0):.0f}%")
    
    st.markdown("---")
    
    # Documents by type
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Documents by Type")
        by_type = stats.get('by_type', {})
        if by_type:
            for doc_type, count in by_type.items():
                st.markdown(f"- **{doc_type.replace('_', ' ').title()}:** {count}")
        else:
            st.info("No documents yet")
    
    with col2:
        st.markdown("### 🔧 System Status")
        st.markdown(f"- **Vector Chunks:** {rag_stats.get('total_chunks', 0)}")
        st.markdown(f"- **Embedding Model:** {rag_stats.get('embedding_model', 'N/A')}")
        
        ollama_status = "✅ Available" if rag_stats.get('ollama_available') else "❌ Not Running"
        st.markdown(f"- **Ollama LLM:** {ollama_status}")
        
        if rag_stats.get('ollama_model'):
            st.markdown(f"- **LLM Model:** {rag_stats.get('ollama_model')}")
    
    st.markdown("---")
    
    # Recent documents
    st.markdown("### 📋 Recent Documents")
    documents = get_all_documents()[:5]
    
    if documents:
        for doc in documents:
            st.markdown(
                f"- **{doc['filename']}** - {doc['doc_type'].replace('_', ' ').title()} "
                f"(Quality: {doc['quality_score']:.0f}%) - {doc['upload_date'][:10]}"
            )
    else:
        st.info("No documents uploaded yet")


def main():
    """Main application entry point"""
    
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Render selected page
    if page == "💬 Chat & Query":
        render_chat_page()
    elif page == "📚 Documents":
        render_documents_page()
    elif page == "📊 Dashboard":
        render_dashboard_page()


if __name__ == "__main__":
    main()

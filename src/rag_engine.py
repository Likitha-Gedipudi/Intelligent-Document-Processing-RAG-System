"""
RAG Engine Module
Retrieval-Augmented Generation using local, free components:
- Sentence Transformers for embeddings (free, local)
- ChromaDB for vector storage (free, local)
- Ollama for LLM inference (free, local)
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, OLLAMA_MODEL, CHROMA_DIR


class RAGEngine:
    """
    RAG Engine that runs entirely locally without any paid APIs
    """
    
    def __init__(self):
        self._embedder = None
        self._chroma_client = None
        self._collection = None
        self._ollama_available = None
    
    @property
    def embedder(self):
        """Lazy load the embedding model"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder
    
    @property
    def collection(self):
        """Lazy load ChromaDB collection"""
        if self._collection is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = self._chroma_client.get_or_create_collection(
                name="banking_documents",
                metadata={"description": "Banking document embeddings"}
            )
        return self._collection
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        if self._ollama_available is not None:
            return self._ollama_available
        
        try:
            import ollama
            # Try to list models
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            # Check if our preferred model or any model is available
            self._ollama_available = any(
                OLLAMA_MODEL.split(':')[0] in name 
                for name in model_names
            ) or len(model_names) > 0
            
            if self._ollama_available and model_names:
                print(f"Ollama available with models: {model_names}")
            
            return self._ollama_available
        except Exception as e:
            print(f"Ollama not available: {e}")
            self._ollama_available = False
            return False
    
    def add_document(
        self,
        doc_id: str,
        chunks: List[Tuple[str, int]],
        metadata: Dict
    ) -> List[str]:
        """
        Add document chunks to the vector database
        
        Args:
            doc_id: Unique document ID
            chunks: List of (chunk_text, chunk_index) tuples
            metadata: Document metadata to store with each chunk
            
        Returns:
            List of embedding IDs
        """
        if not chunks:
            return []
        
        embedding_ids = []
        texts = [chunk[0] for chunk in chunks]
        
        # Generate embeddings for all chunks at once (more efficient)
        embeddings = self.embedder.encode(texts).tolist()
        
        for (chunk_text, chunk_index), embedding in zip(chunks, embeddings):
            embedding_id = f"{doc_id}_chunk_{chunk_index}"
            
            self.collection.add(
                ids=[embedding_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    **metadata,
                    "document_id": doc_id,
                    "chunk_index": chunk_index
                }]
            )
            embedding_ids.append(embedding_id)
        
        return embedding_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_type: str = None
    ) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_doc_type: Optional filter by document type
            
        Returns:
            List of result dictionaries with text, metadata, and score
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build filter if specified
        where_filter = None
        if filter_doc_type:
            where_filter = {"doc_type": filter_doc_type}
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else None
                })
        
        return formatted
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 500
    ) -> str:
        """
        Generate a response using the local Ollama LLM
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            max_tokens: Maximum response length
            
        Returns:
            Generated response text
        """
        if not self.check_ollama_available():
            return self._fallback_response(query, context_chunks)
        
        # Build context from chunks
        context = "\n\n---\n\n".join([
            f"[Source: {chunk['metadata'].get('filename', 'Unknown')}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Build prompt
        prompt = f"""You are a helpful banking document assistant. Answer the user's question based ONLY on the provided context. If the information is not in the context, say "I don't have that information in the uploaded documents."

CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based only on the context above
- Be concise and accurate
- Cite the source document when possible
- If you can't find the answer, say so clearly

ANSWER:"""

        try:
            import ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'num_predict': max_tokens}
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._fallback_response(query, context_chunks)
    
    def _fallback_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Fallback response when Ollama is not available
        Returns relevant context without LLM processing
        """
        if not context_chunks:
            return "No relevant documents found for your query."
        
        response = "**Ollama LLM not available. Here are the relevant document excerpts:**\n\n"
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            filename = chunk['metadata'].get('filename', 'Unknown')
            response += f"**{i}. From {filename}:**\n"
            response += f"{chunk['text'][:500]}...\n\n"
        
        response += "\n*Install and run Ollama to get AI-powered answers: https://ollama.ai*"
        return response
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_doc_type: str = None
    ) -> Dict:
        """
        Full RAG query: search + generate
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            filter_doc_type: Optional filter by document type
            
        Returns:
            Dictionary with answer, sources, and timing info
        """
        start_time = time.time()
        
        # Search for relevant chunks
        results = self.search(question, top_k=top_k, filter_doc_type=filter_doc_type)
        
        search_time = time.time() - start_time
        
        # Generate response
        answer = self.generate_response(question, results)
        
        total_time = time.time() - start_time
        
        # Prepare sources
        sources = []
        seen = set()
        for r in results:
            filename = r['metadata'].get('filename', 'Unknown')
            if filename not in seen:
                sources.append({
                    'filename': filename,
                    'doc_type': r['metadata'].get('doc_type', 'unknown'),
                    'relevance': round(1 - r.get('distance', 0), 3)
                })
                seen.add(filename)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks_retrieved': len(results),
            'search_time_ms': int(search_time * 1000),
            'total_time_ms': int(total_time * 1000)
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'embedding_model': EMBEDDING_MODEL,
                'ollama_available': self.check_ollama_available(),
                'ollama_model': OLLAMA_MODEL if self.check_ollama_available() else None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def delete_document_embeddings(self, doc_id: str) -> bool:
        """Delete all embeddings for a document"""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return False


# Singleton instance
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """Get the singleton RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

"""
Database Module
SQLite-based storage for document metadata and extracted entities
Zero setup required - SQLite is built into Python
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH


def get_connection():
    """Get a database connection with row factory"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            doc_type TEXT,
            upload_date TEXT,
            processed_date TEXT,
            status TEXT DEFAULT 'pending',
            extracted_text TEXT,
            quality_score REAL,
            metadata TEXT
        )
    ''')
    
    # Entities table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            entity_type TEXT,
            entity_value TEXT,
            is_valid INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')
    
    # Document chunks table (for RAG)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding_id TEXT,
            created_at TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')
    
    # Query logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            response_text TEXT,
            sources TEXT,
            execution_time_ms INTEGER,
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def save_document(
    doc_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    doc_type: str,
    extracted_text: str,
    quality_score: float,
    metadata: dict = None
) -> bool:
    """
    Save a processed document to the database
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO documents 
            (id, filename, file_path, file_size, doc_type, upload_date, 
             processed_date, status, extracted_text, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            filename,
            file_path,
            file_size,
            doc_type,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            'completed',
            extracted_text,
            quality_score,
            json.dumps(metadata) if metadata else None
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving document: {e}")
        return False
    finally:
        conn.close()


def save_entities(doc_id: str, entities: Dict[str, List[str]]) -> bool:
    """
    Save extracted entities for a document
    
    Args:
        doc_id: Document ID
        entities: Dictionary of entity_type -> list of values
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for entity_type, values in entities.items():
            for value in values:
                cursor.execute('''
                    INSERT INTO entities (document_id, entity_type, entity_value, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (doc_id, entity_type, value, datetime.now().isoformat()))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving entities: {e}")
        return False
    finally:
        conn.close()


def save_chunks(doc_id: str, chunks: List[tuple], embedding_ids: List[str]) -> bool:
    """
    Save document chunks with their embedding IDs
    
    Args:
        doc_id: Document ID
        chunks: List of (chunk_text, chunk_index) tuples
        embedding_ids: List of embedding IDs from vector DB
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for (chunk_text, chunk_index), embedding_id in zip(chunks, embedding_ids):
            cursor.execute('''
                INSERT INTO document_chunks 
                (document_id, chunk_index, chunk_text, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, chunk_index, chunk_text, embedding_id, datetime.now().isoformat()))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving chunks: {e}")
        return False
    finally:
        conn.close()


def get_document(doc_id: str) -> Optional[Dict]:
    """Get a document by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_documents() -> List[Dict]:
    """Get all documents"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents ORDER BY upload_date DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_document_entities(doc_id: str) -> Dict[str, List[str]]:
    """Get all entities for a document"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT entity_type, entity_value 
        FROM entities 
        WHERE document_id = ?
    ''', (doc_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    entities = {}
    for row in rows:
        entity_type = row['entity_type']
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(row['entity_value'])
    
    return entities


def get_documents_by_type(doc_type: str) -> List[Dict]:
    """Get all documents of a specific type"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents WHERE doc_type = ?', (doc_type,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def search_entities(entity_type: str = None, value_pattern: str = None) -> List[Dict]:
    """
    Search for entities across all documents
    
    Args:
        entity_type: Filter by entity type (e.g., 'pan_number')
        value_pattern: SQL LIKE pattern for value search
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = 'SELECT e.*, d.filename FROM entities e JOIN documents d ON e.document_id = d.id WHERE 1=1'
    params = []
    
    if entity_type:
        query += ' AND e.entity_type = ?'
        params.append(entity_type)
    
    if value_pattern:
        query += ' AND e.entity_value LIKE ?'
        params.append(f'%{value_pattern}%')
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def log_query(query_text: str, response_text: str, sources: List[str], execution_time_ms: int):
    """Log a query for analytics"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO query_logs (query_text, response_text, sources, execution_time_ms, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (query_text, response_text, json.dumps(sources), execution_time_ms, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_statistics() -> Dict:
    """Get database statistics for dashboard"""
    conn = get_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total documents
    cursor.execute('SELECT COUNT(*) as count FROM documents')
    stats['total_documents'] = cursor.fetchone()['count']
    
    # Documents by type
    cursor.execute('SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type')
    stats['by_type'] = {row['doc_type']: row['count'] for row in cursor.fetchall()}
    
    # Average quality score
    cursor.execute('SELECT AVG(quality_score) as avg_score FROM documents')
    result = cursor.fetchone()
    stats['avg_quality_score'] = round(result['avg_score'], 2) if result['avg_score'] else 0
    
    # Total entities
    cursor.execute('SELECT COUNT(*) as count FROM entities')
    stats['total_entities'] = cursor.fetchone()['count']
    
    # Total queries
    cursor.execute('SELECT COUNT(*) as count FROM query_logs')
    stats['total_queries'] = cursor.fetchone()['count']
    
    conn.close()
    return stats


def delete_document(doc_id: str) -> bool:
    """Delete a document and its related data"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM entities WHERE document_id = ?', (doc_id,))
        cursor.execute('DELETE FROM document_chunks WHERE document_id = ?', (doc_id,))
        cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
    finally:
        conn.close()


# Initialize database on module import
init_database()

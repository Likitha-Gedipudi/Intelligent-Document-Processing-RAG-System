"""
Document Processing Module
Handles PDF extraction, text processing, and document classification
All operations are free and local - no paid APIs
"""
import re
from pathlib import Path
from typing import List, Tuple
import PyPDF2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2 (free, local)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
    except Exception as e:
        raise Exception(f"Error extracting PDF: {str(e)}")
    
    return text.strip()


def extract_text_from_txt(file_path: str) -> str:
    """
    Read text from a plain text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File contents as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word document
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as a string
    """
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise Exception("python-docx not installed. Run: pip install python-docx")


def extract_text(file_path: str) -> str:
    """
    Extract text from any supported file type
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text as a string
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.txt':
        return extract_text_from_txt(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def classify_document(text: str) -> str:
    """
    Classify document type using keyword matching (free, no ML model needed)
    
    Args:
        text: Document text content
        
    Returns:
        Document type classification
    """
    text_lower = text.lower()
    
    # Keywords for each document type
    classifications = {
        'loan_application': [
            'loan application', 'loan amount', 'emi', 'interest rate',
            'home loan', 'personal loan', 'business loan', 'loan tenure',
            'principal amount', 'loan purpose', 'collateral'
        ],
        'kyc_document': [
            'aadhaar', 'aadhar', 'pan card', 'kyc', 'know your customer',
            'identity proof', 'address proof', 'passport', 'voter id',
            'driving license', 'verification', 'identity document'
        ],
        'bank_statement': [
            'account statement', 'transaction history', 'opening balance',
            'closing balance', 'credit', 'debit', 'statement period',
            'account number', 'transaction date', 'bank statement'
        ],
        'salary_slip': [
            'salary slip', 'pay slip', 'gross salary', 'net salary',
            'basic pay', 'allowances', 'deductions', 'pf contribution',
            'income tax', 'take home', 'earnings'
        ]
    }
    
    # Count keyword matches for each type
    scores = {}
    for doc_type, keywords in classifications.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score
    
    # Get the type with highest score
    best_type = max(scores, key=scores.get)
    
    # If no keywords matched, return 'other'
    if scores[best_type] == 0:
        return 'other'
    
    return best_type


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int]]:
    """
    Split text into overlapping chunks for embedding
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of (chunk_text, chunk_index) tuples
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= chunk_size:
        return [(text, 0)]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation
            for i in range(end, max(start + chunk_size // 2, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, chunk_index))
            chunk_index += 1
        
        start = end - overlap
    
    return chunks


def get_document_stats(text: str) -> dict:
    """
    Get basic statistics about a document
    
    Args:
        text: Document text content
        
    Returns:
        Dictionary with document statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'page_markers': text.count('--- Page'),
    }

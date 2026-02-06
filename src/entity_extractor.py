"""
Entity Extraction Module
Regex-based Named Entity Recognition for Indian banking documents
All operations are free and local - no paid NLP services needed
"""
import re
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata"""
    entity_type: str
    value: str
    confidence: float = 1.0  # Regex matches are deterministic


# Regex patterns for Indian banking entities
PATTERNS = {
    # Indian PAN Card: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
    'pan_number': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    
    # Indian Aadhaar: 12 digits, optionally with spaces (e.g., 1234 5678 9012)
    'aadhaar_number': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    
    # Indian mobile numbers: Start with 6-9, followed by 9 digits
    'phone_number': r'\b[6-9]\d{9}\b',
    
    # Email addresses
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    
    # Indian currency amounts (with ₹ or Rs. prefix)
    'amount': r'(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{2})?',
    
    # Dates in common formats (DD/MM/YYYY, DD-MM-YYYY)
    'date': r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
    
    # Bank account numbers (9-18 digits)
    'account_number': r'\b\d{9,18}\b',
    
    # IFSC codes: 4 letters + 0 + 6 alphanumeric (e.g., SBIN0001234)
    'ifsc_code': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
    
    # Indian PIN codes (6 digits, first digit 1-9)
    'pin_code': r'\b[1-9]\d{5}\b',
    
    # Percentage values
    'percentage': r'\b\d+(?:\.\d+)?%',
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract all entities from text using regex patterns
    
    Args:
        text: Document text content
        
    Returns:
        Dictionary mapping entity types to lists of found values
    """
    entities = {}
    
    for entity_type, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Remove duplicates while preserving order
        unique_matches = list(dict.fromkeys(matches))
        if unique_matches:
            entities[entity_type] = unique_matches
    
    return entities


def extract_entities_with_positions(text: str) -> List[Dict]:
    """
    Extract entities with their positions in the text
    
    Args:
        text: Document text content
        
    Returns:
        List of dictionaries with entity info and positions
    """
    results = []
    
    for entity_type, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append({
                'type': entity_type,
                'value': match.group(),
                'start': match.start(),
                'end': match.end(),
                'context': text[max(0, match.start()-30):min(len(text), match.end()+30)]
            })
    
    return sorted(results, key=lambda x: x['start'])


def validate_pan(pan: str) -> bool:
    """
    Validate PAN number format
    First 3 chars: Alphabetic series (AAA-ZZZ)
    4th char: Status of holder (P=Person, C=Company, etc.)
    5th char: First letter of surname/name
    Next 4: Sequential numbers (0001-9999)
    Last: Alphabetic check digit
    """
    if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan):
        return False
    
    # 4th character validation (holder type)
    valid_4th = ['P', 'C', 'H', 'F', 'A', 'T', 'B', 'L', 'J', 'G']
    if pan[3] not in valid_4th:
        return False
    
    return True


def validate_aadhaar(aadhaar: str) -> bool:
    """
    Validate Aadhaar number (basic validation)
    - Must be 12 digits
    - Cannot start with 0 or 1
    """
    # Remove spaces
    aadhaar_clean = aadhaar.replace(' ', '')
    
    if not re.match(r'^\d{12}$', aadhaar_clean):
        return False
    
    # First digit cannot be 0 or 1
    if aadhaar_clean[0] in ['0', '1']:
        return False
    
    return True


def validate_ifsc(ifsc: str) -> bool:
    """
    Validate IFSC code format
    - 11 characters
    - First 4: Bank code (alphabets)
    - 5th: Always 0
    - Last 6: Branch code (alphanumeric)
    """
    return bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', ifsc))


def get_entity_summary(entities: Dict[str, List[str]]) -> Dict:
    """
    Get a summary of extracted entities with counts and validation
    
    Args:
        entities: Dictionary of extracted entities
        
    Returns:
        Summary dictionary with counts and validated entities
    """
    summary = {
        'total_entities': sum(len(v) for v in entities.values()),
        'entity_counts': {k: len(v) for k, v in entities.items()},
        'validated': {}
    }
    
    # Validate specific entity types
    if 'pan_number' in entities:
        summary['validated']['pan_numbers'] = [
            {'value': pan, 'valid': validate_pan(pan)} 
            for pan in entities['pan_number']
        ]
    
    if 'aadhaar_number' in entities:
        summary['validated']['aadhaar_numbers'] = [
            {'value': aadhaar, 'valid': validate_aadhaar(aadhaar)} 
            for aadhaar in entities['aadhaar_number']
        ]
    
    if 'ifsc_code' in entities:
        summary['validated']['ifsc_codes'] = [
            {'value': ifsc, 'valid': validate_ifsc(ifsc)} 
            for ifsc in entities['ifsc_code']
        ]
    
    return summary


def calculate_quality_score(text: str, doc_type: str) -> float:
    """
    Calculate a data quality score based on extracted entities
    
    Args:
        text: Document text
        doc_type: Type of document
        
    Returns:
        Quality score from 0 to 100
    """
    entities = extract_entities(text)
    
    # Required entities by document type
    required_entities = {
        'loan_application': ['pan_number', 'phone_number', 'amount', 'date'],
        'kyc_document': ['pan_number', 'aadhaar_number', 'phone_number', 'date'],
        'bank_statement': ['account_number', 'ifsc_code', 'date', 'amount'],
        'salary_slip': ['pan_number', 'amount', 'date'],
        'other': []
    }
    
    required = required_entities.get(doc_type, [])
    
    if not required:
        return 75.0  # Default score for 'other' documents
    
    # Calculate completeness score
    found = sum(1 for r in required if r in entities)
    completeness = (found / len(required)) * 100
    
    # Bonus for validated entities
    validation_bonus = 0
    if 'pan_number' in entities:
        valid_pans = sum(1 for pan in entities['pan_number'] if validate_pan(pan))
        if valid_pans > 0:
            validation_bonus += 5
    
    if 'aadhaar_number' in entities:
        valid_aadhaar = sum(1 for a in entities['aadhaar_number'] if validate_aadhaar(a))
        if valid_aadhaar > 0:
            validation_bonus += 5
    
    return min(100, completeness + validation_bonus)

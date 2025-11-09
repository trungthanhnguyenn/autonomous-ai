# perception/text/nlp_utils.py
"""Deterministic NLP utilities (no LLM)"""

from typing import List, Dict
import re

def extract_intent(text: str) -> str:
    """
    Extract user intent from text
    Paper: Part of semantic understanding
    """
    text_lower = text.lower()
    
    intent_patterns = {
        "search": ["search", "find", "look for", "query", "lookup"],
        "navigate": ["go to", "open", "visit", "click on", "access"],
        "execute": ["run", "execute", "perform", "do", "start"],
        "analyze": ["analyze", "check", "examine", "review", "assess"],
        "create": ["create", "make", "build", "write", "generate"],
        "delete": ["delete", "remove", "clear", "erase"],
        "update": ["update", "change", "modify", "edit", "alter"]
    }
    
    for intent, keywords in intent_patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                return intent
    
    return "query"

def extract_entities(text: str) -> List[Dict]:
    """Extract named entities"""
    entities = []
    
    # Pattern-based extraction
    patterns = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "URL": r'http[s]?://\S+',
        "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "DATE": r'\d{1,2}/\d{1,2}/\d{4}',
        "NUMBER": r'\b\d+\b'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "type": entity_type,
                "value": match.group(),
                "span": (match.start(), match.end())
            })
    
    return entities

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
    }
    
    words = text.lower().split()
    keywords = [w.strip(',.!?;:') for w in words 
                if w.lower() not in stopwords and len(w) > 2]
    
    return list(set(keywords))

def assess_complexity(text: str) -> str:
    """Assess complexity"""
    word_count = len(text.split())
    
    if word_count < 10:
        return "simple"
    elif word_count < 50:
        return "moderate"
    else:
        return "complex"

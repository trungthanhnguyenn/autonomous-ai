# perception/text/text_perception.py
"""
Paper Section 3.1: Text-Based Perception
"Direct text descriptions where the environment communicates 
state to the LLM through natural language."
"""

from typing import Any, List, Dict
from core.agent_context import AgentContext, PerceptualData, PerceptionModality
from perception.base import PerceptionSystem
from perception.text.nlp_utils import (
    extract_intent,
    extract_entities, 
    extract_keywords,
    assess_complexity
)
import re
import time

class TextPerceptionSystem(PerceptionSystem):
    """
    Paper Section 3.1: Text-Based Perception
    
    Approach: Process text input through NLP pipeline
    - No LLM needed (deterministic)
    - Fast & reliable
    """
    
    def __init__(self):
        super().__init__()
        self.supported_modalities = [PerceptionModality.TEXT]
        self._current_input_text = None  # Store current input for semantic representation
    
    def perceive(self, raw_input: Any, context: AgentContext) -> AgentContext:
        """
        Main perception pipeline for text
        
        Steps (Paper-aligned):
        1. Object Recognition: Extract entities
        2. Feature Extraction: Extract intent, keywords
        3. Semantic Representation: Build meaning
        """
        start_time = time.time()
        context.log_step("TextPerception", "Starting text perception")
        
        # Validate input
        if not isinstance(raw_input, str):
            raw_input = str(raw_input)
        
        # Store for semantic representation
        self._current_input_text = raw_input
        
        # === STEP 1: Object Recognition ===
        objects = self.extract_objects(raw_input)
        context.log_step("TextPerception", f"Extracted {len(objects)} objects")
        
        # === STEP 2: Feature Extraction ===
        features = self.extract_features(objects)
        context.log_step("TextPerception", f"Extracted features: {list(features.keys())}")
        
        # === STEP 3: Semantic Representation ===
        semantic_rep = self.build_semantic_representation(objects, features)
        context.log_step("TextPerception", f"Built semantic representation")
        
        # Create perception data directly
        context.perception = PerceptualData(
            modality=PerceptionModality.TEXT,
            raw_input=raw_input,
            extracted_objects=objects,
            semantic_representation=semantic_rep,
            confidence_scores=self._compute_confidence(objects, features),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        return context
    
    def extract_objects(self, input_data: str) -> List[Dict]:
        """
        Paper Step 1: Object Recognition
        Extract named entities (entities that perception recognizes)
        
        Uses centralized entity extraction from nlp_utils to avoid duplication.
        """
        # Use the centralized entity extraction which already handles
        # EMAIL, URL, PHONE, DATE, NUMBER patterns
        objects = extract_entities(input_data)
        
        return objects
    
    def extract_features(self, objects: List[Dict]) -> Dict:
        """
        Paper Step 2: Feature Extraction
        Extract meaningful features from recognized objects
        """
        return {
            "object_count": len(objects),
            "object_types": list(set(o["type"] for o in objects)),
            "has_email": any(o["type"] == "EMAIL" for o in objects),
            "has_url": any(o["type"] == "URL" for o in objects)
        }
    
    def build_semantic_representation(self, 
                                     objects: List[Dict],
                                     features: Dict) -> Dict:
        """
        Paper Step 3: Semantic Representation
        Build meaningful interpretation of environment
        
        Uses the stored input text from the current perception cycle.
        """
        input_text = self._current_input_text or ""
        intent = extract_intent(input_text)
        keywords = extract_keywords(input_text)
        
        return {
            "intent": intent,
            "keywords": keywords,
            "features": features,
            "entity_count": len(objects),
            "complexity": assess_complexity(input_text),
            "has_command": any(kw in intent.lower() for kw in ["search", "navigate", "execute"])
        }
    
    def _compute_confidence(self, objects: List[Dict], features: Dict) -> Dict:
        """Confidence scores for different aspects"""
        return {
            "entity_recognition": 0.95 if objects else 0.5,
            "overall": 0.8
        }

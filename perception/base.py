# perception/base.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from core.agent_context import AgentContext

class PerceptionModality(Enum):
    """Paper Section 3: Perception modalities"""
    TEXT = "text"                  # Text-based perception
    VISION = "vision"              # Vision/VLM perception
    STRUCTURED = "structured"      # A11y, HTML, JSON
    SENSOR = "sensor"              # Real-world sensors
    TOOL_AUGMENTED = "tool"        # APIs, external tools

@dataclass
class PerceptionOutput:
    """
    Paper: "Perception converts environmental percepts 
    into meaningful representations"
    
    This is the OUTPUT of perception system
    """
    modality: PerceptionModality
    raw_input: Any                      # Original input
    extracted_objects: List[Dict]       # Recognized entities/objects
    semantic_representation: Dict       # Meaningful interpretation
    confidence_scores: Dict[str, float] # Confidence per object
    metadata: Dict = None               # Processing details
    processing_time_ms: float = 0.0
    
    def to_context_format(self):
        """Convert to AgentContext perception data"""
        return {
            "modality": self.modality.value,
            "objects": self.extracted_objects,
            "semantic": self.semantic_representation,
            "confidence": self.confidence_scores
        }
    
    def to_dict(self):
        """Convert to dict (for compatibility)"""
        return {
            "modality": self.modality.value,
            "raw_input": str(self.raw_input)[:200],
            "extracted_objects": self.extracted_objects,
            "semantic_representation": self.semantic_representation,
            "confidence_scores": self.confidence_scores,
            "processing_time_ms": self.processing_time_ms
        }
    
    def get_confidence(self, component: str = "overall") -> float:
        """Get confidence score (for compatibility)"""
        return self.confidence_scores.get(component, 0.0)

class PerceptionSystem(ABC):
    """
    Abstract Perception System
    
    Paper Definition:
    "A perception system that converts environmental percepts 
    into meaningful representations. Key components include 
    object recognition, feature extraction, and contextual 
    understanding of the environment."
    """
    
    supported_modalities: List[PerceptionModality] = []
    
    @abstractmethod
    def perceive(self, 
                 raw_input: Any, 
                 context: AgentContext) -> AgentContext:
        """
        Main perception method
        
        Converts raw environmental input into structured 
        representations that reasoning system can understand.
        
        Args:
            raw_input: Raw data from environment
            context: Current agent context
            
        Returns:
            Updated context with perception data
        """
        pass
    
    @abstractmethod
    def extract_objects(self, input_data: Any) -> List[Dict]:
        """
        Extract recognizable objects/entities from input
        Paper: Object Recognition phase
        """
        pass
    
    @abstractmethod
    def extract_features(self, objects: List[Dict]) -> Dict:
        """
        Extract meaningful features from objects
        Paper: Feature Extraction phase
        """
        pass
    
    @abstractmethod
    def build_semantic_representation(self, 
                                     objects: List[Dict],
                                     features: Dict) -> Dict:
        """
        Build semantic understanding of environment
        Paper: Contextual Understanding phase
        """
        pass
    
    def supports(self, modality: PerceptionModality) -> bool:
        return modality in self.supported_modalities


class PerceptionPipeline:
    """
    Paper Section 3.4: "Combining Multiple Perception Modalities"
    
    Manages multiple percevers working together
    """
    
    def __init__(self):
        self.perceptors: Dict[PerceptionModality, PerceptionSystem] = {}
        self.pipeline_order: List[PerceptionModality] = []
    
    def register_perceptor(self, 
                          modality: PerceptionModality,
                          perceptor: PerceptionSystem):
        """Register a perceiver for specific modality"""
        self.perceptors[modality] = perceptor
        self.pipeline_order.append(modality)
    
    def perceive_multimodal(self,
                           inputs: Dict[PerceptionModality, Any],
                           context: AgentContext) -> AgentContext:
        """
        Process multiple modalities in sequence
        Each perceptor enriches context for next one
        """
        for modality in self.pipeline_order:
            if modality in inputs:
                perceptor = self.perceptors[modality]
                context = perceptor.perceive(inputs[modality], context)
        
        return context
    
    def perceive_adaptive(self,
                         inputs: Dict[PerceptionModality, Any],
                         context: AgentContext) -> AgentContext:
        """
        Adaptive perception: choose perceptors based on context
        E.g., if confidence is low, try different modality
        """
        for modality, input_data in inputs.items():
            if modality in self.perceptors:
                context = self.perceptors[modality].perceive(
                    input_data, context
                )
                # Check confidence
                if context.perception["confidence"] < 0.5:
                    # Try alternative perceiver if available
                    pass
        
        return context

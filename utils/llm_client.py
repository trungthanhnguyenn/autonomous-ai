"""
LLM Client Wrapper
==================

Provides unified interface for LLM and VLM using OpenAI-compatible endpoints.
Reads configuration from .env file.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient:
    """
    LLM Client for text reasoning
    Uses custom OpenAI-compatible endpoint from .env
    """
    
    def __init__(self):
        """Initialize LLM client with .env configuration"""
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
        
        if not self.api_key or not self.base_url:
            raise ValueError(
                "Missing API_KEY or BASE_URL in .env file. "
                "Please set: API_KEY, BASE_URL, LLM_MODEL_NAME"
            )
        
        # Initialize OpenAI client with custom endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✓ LLM Client initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Endpoint: {self.base_url}")
    
    def create(self, messages: List[Dict], **kwargs) -> Any:
        """
        Create completion (OpenAI-compatible interface)
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            OpenAI response object
        """
        # Set defaults
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        
        except Exception as e:
            print(f"⚠ LLM API Error: {e}")
            raise
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        """
        Generate text completion (simplified interface)
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
        
        Returns:
            Generated text content
        """
        response = self.create(messages=messages, **kwargs)
        return response.choices[0].message.content


class VLMClient:
    """
    VLM (Vision-Language Model) Client
    Uses OpenRouter endpoint from .env for multimodal tasks
    """
    
    def __init__(self):
        """Initialize VLM client with .env configuration"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = os.getenv("VLM_MODEL_NAME")
        
        if not self.api_key or not self.base_url or not self.model_name:
            raise ValueError(
                "Missing VLM configuration in .env file. "
                "Please set: OPENAI_API_KEY, OPENAI_BASE_URL, VLM_MODEL"
            )
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✓ VLM Client initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Endpoint: {self.base_url}")
    
    def create(self, messages: List[Dict], **kwargs) -> Any:
        """
        Create completion with vision support
        
        Args:
            messages: List of message dicts (can include images)
            **kwargs: Additional parameters
        
        Returns:
            OpenAI response object
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        
        except Exception as e:
            print(f"⚠ VLM API Error: {e}")
            raise
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        """
        Generate text from text or image inputs
        
        Args:
            messages: List of message dicts
            **kwargs: Additional parameters
        
        Returns:
            Generated text content
        """
        response = self.create(messages=messages, **kwargs)
        return response.choices[0].message.content
    
    def analyze_image(self, image_url: str, prompt: str = "What do you see in this image?") -> str:
        """
        Analyze an image and return description
        
        Args:
            image_url: URL or base64 image
            prompt: Question about the image
        
        Returns:
            Image analysis text
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        return self.generate(messages=messages)


def create_llm_client() -> LLMClient:
    """Factory function to create LLM client"""
    return LLMClient()


def create_vlm_client() -> VLMClient:
    """Factory function to create VLM client"""
    return VLMClient()


def test_clients():
    """Test both clients with simple queries"""
    print("\n" + "="*70)
    print("TESTING LLM AND VLM CLIENTS")
    print("="*70 + "\n")
    
    # Test LLM
    print("1. Testing LLM Client...")
    try:
        llm = create_llm_client()
        response = llm.generate(
            messages=[{"role": "user", "content": "Say 'Hello, I am working!' in one sentence."}],
            temperature=0.5,
            max_tokens=50
        )
        print(f"   Response: {response}")
        print("   ✓ LLM Client working!\n")
    except Exception as e:
        print(f"   ✗ LLM Client failed: {e}\n")
    
    # Test VLM
    print("2. Testing VLM Client...")
    try:
        vlm = create_vlm_client()
        response = vlm.generate(
            messages=[{"role": "user", "content": "Say 'Hello from VLM!' in one sentence."}],
            temperature=0.5,
            max_tokens=50
        )
        print(f"   Response: {response}")
        print("   ✓ VLM Client working!\n")
    except Exception as e:
        print(f"   ✗ VLM Client failed: {e}\n")
    
    print("="*70)
    print("CLIENT TESTS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test the clients
    test_clients()

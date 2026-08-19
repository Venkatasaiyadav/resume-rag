"""
LLM_CLIENT.PY - Communicate with Groq LLM

WHY GROQ?
═════════

Groq is a specialized AI inference platform that runs open-source LLMs
on custom LPU (Language Processing Unit) hardware.

BENEFITS:
✅ FREE tier (generous limits: ~30 requests/min)
✅ FASTEST inference in the world (500+ tokens/second!)
✅ Multiple model options (Llama 3, Mixtral, Gemma)
✅ OpenAI-compatible API
✅ No credit card required

MODELS AVAILABLE:
- openai/gpt-oss-120b: Best quality (120 billion parameters)
- openai/gpt-oss-20b: Fast, still good quality
- qwen/qwen3.6-27b: Good alternative

WE USE: openai/gpt-oss-120b (best quality, still very fast)
"""

from groq import Groq
from typing import Optional
from app.config import settings


class LLMClient:
    """
    Client for Groq API (uses openai/gpt-oss-120b by default).
    """
    
    def __init__(self):
        """Initialize Groq client with API key"""
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set! Get one FREE at: "
                "https://console.groq.com/keys"
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        
        print(f"🤖 LLM initialized: {self.model} (via Groq)")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Complete prompt (system + context + question)
            
        Returns:
            Generated text response
            
        FLOW:
        prompt → Groq API → GPT-OSS 120B → Generated answer
        
        GROQ API STRUCTURE (OpenAI-compatible):
        - Uses chat completions format
        - messages: list of {role, content}
        - role can be: "system", "user", "assistant"
        """
        try:
            # Send request to Groq
            # We use chat completions format (OpenAI-style)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                top_p=0.8,
                stream=False,  # We want full response, not streaming
            )
            
            # Extract the generated text
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                return answer if answer else "No response generated."
            else:
                return "I couldn't generate a response. Please try rephrasing."
                
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            print(f"  ❌ {error_msg}")
            return error_msg
    
    def generate_with_metadata(self, prompt: str) -> dict:
        """
        Generate response with additional metadata.
        Useful for debugging and Postman testing.
        
        Returns detailed info about the LLM call:
        - answer: The generated text
        - tokens used
        - model info
        - finish reason
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                top_p=0.8,
                stream=False,
            )
            
            # Extract detailed info
            choice = response.choices[0] if response.choices else None
            answer = choice.message.content if choice else "No response"
            
            return {
                "answer": answer,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "finish_reason": choice.finish_reason if choice else "unknown",
                "model": self.model,
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "finish_reason": "error",
                "model": self.model,
            }
    
    def generate_with_system_prompt(self, system_prompt: str, user_query: str) -> str:
        """
        Alternative method using separate system + user prompts.
        This is the "proper" way to structure chat with LLMs.
        
        Args:
            system_prompt: Instructions for the LLM (persona, rules)
            user_query: The actual user question with context
            
        Returns:
            Generated response
            
        WHY SEPARATE?
        - LLMs are trained to give more weight to system prompts
        - Cleaner separation of instructions vs data
        - Better for controlling behavior
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            
            return response.choices[0].message.content if response.choices else "No response"
            
        except Exception as e:
            return f"Error: {str(e)}"


# Test standalone
if __name__ == "__main__":
    llm = LLMClient()
    
    # Simple test
    response = llm.generate("What is Python programming language? Answer in 2 sentences.")
    print(f"\n🤖 Response:\n{response}")
    
    # With metadata
    print("\n" + "="*50)
    result = llm.generate_with_metadata("Explain RAG in 2 sentences.")
    print(f"Answer: {result['answer']}")
    print(f"Tokens used: {result['total_tokens']}")
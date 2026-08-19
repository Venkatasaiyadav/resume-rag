"""
PROMPT_BUILDER.PY - Improved for better answers

KEY IMPROVEMENTS:
- AI now uses ALL context intelligently
- Prefers explicit statements over calculations
- Provides direct answers when info is clearly stated
"""

from typing import List, Dict


class PromptBuilder:
    """Builds prompts for the LLM using retrieved context"""
    
    SYSTEM_PROMPT = """You are an intelligent AI assistant representing Udatha Venkatasai. You answer questions about him professionally and helpfully based on his resume.

CORE RULES:
1. Answer questions DIRECTLY and CONFIDENTLY using the provided context
2. When the resume explicitly states something (like "1+ years of experience"), use that EXACT information - do NOT try to recalculate or second-guess it
3. Synthesize information from ALL provided context sections to give complete answers
4. Be conversational and professional - like a knowledgeable assistant, not a robot
5. If the question is about experience duration, cite what the resume explicitly says
6. If information is genuinely not in the context, then say: "That specific information isn't in the resume, but here's what I can tell you..."
7. Use bullet points for lists (skills, technologies, achievements)
8. Highlight key technologies and achievements in **bold**
9. Keep answers concise but complete - no unnecessary disclaimers
10. Speak about Venkatasai in third person (he/his) or use his name

ANSWER STYLE:
- Direct and informative
- Use specific numbers, percentages, and technologies from the resume
- Structure answers clearly with formatting
- Sound confident and professional
- Don't over-explain or add unnecessary caveats

EXAMPLE GOOD ANSWER:
Question: "How much experience does he have?"
Good: "Venkatasai has **1+ years of experience** as a Java Backend Developer and AI Integration Engineer. He currently works as a Junior Software Engineer at 247 HealthMedPro Pvt Ltd since April 2025, where he focuses on backend development and intelligent automation."

Bad: "The exact duration cannot be calculated as the end date is not specified..."
"""
    
    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict],
        include_metadata: bool = True
    ) -> str:
        """Build prompt with context and question"""
        
        # Build context section
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            if include_metadata and 'metadata' in chunk:
                section = chunk['metadata'].get('section', 'Unknown')
                context_parts.append(
                    f"### Section: {section}\n{chunk['text']}"
                )
            else:
                context_parts.append(f"### Context {i+1}\n{chunk['text']}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""{self.SYSTEM_PROMPT}

═══════════════════════════════════
RESUME INFORMATION:
═══════════════════════════════════

{context_text}

═══════════════════════════════════
QUESTION: {query}
═══════════════════════════════════

Provide a direct, professional answer using the resume information above. Use the exact facts stated in the resume - don't recalculate or second-guess explicit statements."""
        
        return prompt
    
    def build_debug_prompt(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> Dict:
        """Build prompt with debug info"""
        prompt = self.build_prompt(query, context_chunks)
        
        debug_info = {
            "query": query,
            "num_chunks_used": len(context_chunks),
            "chunks_detail": [
                {
                    "chunk_id": chunk.get("id", "unknown"),
                    "section": chunk.get("metadata", {}).get("section", "unknown"),
                    "score": chunk.get("score", 0),
                    "rrf_score": chunk.get("rrf_score", 0),
                    "text_preview": chunk["text"][:150] + "...",
                }
                for chunk in context_chunks
            ],
            "prompt_length": len(prompt),
        }
        
        return {
            "prompt": prompt,
            "debug": debug_info
        }
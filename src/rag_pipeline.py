from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import logging
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

class ResearchRAGPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.vector_store = FAISSVectorStore()
        
    def add_papers(self, documents):
        """Add research papers to the knowledge base"""
        self.vector_store.add_documents(documents)
    
    def create_research_context(self, retrieved_docs: List[Tuple], query: str) -> str:
        """Create context from retrieved documents"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs):
            metadata = doc.metadata
            source_info = f"Paper: {metadata.get('title', 'Unknown')} | Section: {metadata.get('section', 'Unknown')} | Authors: {metadata.get('authors', 'Unknown')}"
            
            context_parts.append(f"""
Source {i+1}: {source_info}
Content: {doc.page_content}
Relevance Score: {score:.3f}
---""")
        
        return "\n".join(context_parts)
    
    def generate_research_prompt(self, query: str, context: str) -> str:
        """Generate specialized prompt for research queries"""
        return f"""You are an AI research assistant specializing in academic paper analysis. 
Your role is to help researchers understand, compare, and analyze academic papers.

INSTRUCTIONS:
- Provide detailed, accurate answers based on the provided context
- Always cite which paper and section your information comes from
- When multiple papers discuss the same topic, compare their approaches
- Highlight any contradictions or different perspectives between papers
- If discussing methodologies, be specific about techniques, datasets, and metrics
- Identify research gaps or opportunities when relevant
- Use academic language appropriate for researchers

CONTEXT FROM RESEARCH PAPERS:
{context}

RESEARCH QUESTION: {query}

DETAILED RESEARCH-FOCUSED ANSWER:"""

    def query(self, question: str, k: int = 5, filter_section: str = None) -> Dict:
        """Query the research paper knowledge base"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                question, k=k, filter_section=filter_section
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information in the uploaded papers to answer your question. Please try rephrasing your question or upload more relevant papers.",
                    "sources": [],
                    "retrieved_chunks": 0
                }
            
            # Create context
            context = self.create_research_context(retrieved_docs, question)
            
            # Generate prompt
            prompt = self.generate_research_prompt(question, context)
            
            # Generate response
            config = types.GenerateContentConfig(
                max_output_tokens=800,
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.8
            )
            
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=config
            )
            
            # Extract sources
            sources = []
            for doc, score in retrieved_docs:
                source_info = {
                    'title': doc.metadata.get('title', 'Unknown Title'),
                    'section': doc.metadata.get('section', 'Unknown'),
                    'authors': doc.metadata.get('authors', 'Unknown Authors'),
                    'year': doc.metadata.get('year', 'Unknown'),
                    'relevance_score': round(score, 3),
                    'content_preview': doc.page_content[:200] + "..."
                }
                sources.append(source_info)
            
            return {
                "answer": response.text,
                "sources": sources,
                "retrieved_chunks": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "retrieved_chunks": 0
            }
    
    def get_papers_overview(self) -> Dict:
        """Get overview of papers in the knowledge base"""
        return self.vector_store.get_papers_summary()
    
    def clear_knowledge_base(self):
        """Clear all papers from the knowledge base"""
        self.vector_store.clear()
    
    def suggest_research_questions(self) -> List[str]:
        """Suggest relevant research questions based on loaded papers"""
        papers_summary = self.get_papers_overview()
        if not papers_summary:
            return []
        
        # Generate suggestions based on common research patterns
        suggestions = [
            "What methodologies are used across these papers?",
            "Compare the evaluation metrics used in these studies",
            "What datasets were used for experiments?",
            "What are the main contributions of each paper?",
            "Identify any research gaps or future work mentioned",
            "Compare the performance results across papers",
            "What are the common limitations mentioned?",
            "How do the related work sections compare?",
        ]
        
        return suggestions
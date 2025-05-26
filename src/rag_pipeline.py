from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import logging
from .vector_store import FAISSVectorStore
from config import AppConfig
import time

logger = logging.getLogger(__name__)

class ResearchRAGPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.vector_store = FAISSVectorStore()
        self._cached_suggestions = None
        self._knowledge_base_version = 0
        self.config = AppConfig()

    def add_papers(self, documents: List) -> None:
        """Add research papers to the knowledge base"""
        self.vector_store.add_documents(documents)
        self._knowledge_base_version += 1
        self._cached_suggestions = None
        logger.info(f"Added {len(documents)} documents to knowledge base")

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
Content: {doc.page_content[:200]}...
Relevance Score: {score:.3f}
---""")
        
        return "\n".join(context_parts)

    def generate_research_prompt(self, query: str, context: str) -> str:
        """Generate prompt for research queries"""
        return f"""You are an AI research assistant for academic paper analysis.
INSTRUCTIONS:
- Answer based on the provided context from research papers.
- Cite the paper title, section, and authors for each piece of information.
- Compare approaches if multiple papers address the query.
- Highlight contradictions or differing perspectives.
- Use specific details for methodologies, datasets, and metrics.
- Identify research gaps when relevant.
- Use clear, academic language.

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""

    def query(self, question: str, k: int = 5, filter_section: str = None, stream: bool = False) -> Dict:
        """Query the research paper knowledge base with retry logic and optional streaming"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                retrieved_docs = self.vector_store.similarity_search(
                    query=question, k=k, filter_section=filter_section
                )

                if not retrieved_docs:
                    return {
                        "answer": "No relevant information found. Try rephrasing or uploading more papers.",
                        "sources": [],
                        "retrieved_chunks": 0
                    }

                context = self.create_research_context(retrieved_docs, question)
                prompt = self.generate_research_prompt(question, context)

                config = types.GenerateContentConfig(
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=float(self.config.temperature),
                    top_p=float(self.config.top_p)
                )

                if stream:
                    response = self.client.models.generate_content_stream(
                        model="gemma-3-12b-it",
                        contents=[prompt],
                        config=config
                    )
                    answer = ""
                    for chunk in response:
                        if chunk.text:
                            answer += chunk.text
                else:
                    response = self.client.models.generate_content(
                        model="gemma-3-12b-it",
                        contents=[prompt],
                        config=config
                    )
                    answer = response.text

                sources = [
                    {
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'section': doc.metadata.get('section', 'Unknown'),
                        'authors': doc.metadata.get('authors', 'Unknown Authors'),
                        'year': doc.metadata.get('year', 'Unknown'),
                        'relevance_score': round(float(score), 3),
                        'content_preview': doc.page_content[:200] + "..."
                    } for doc, score in retrieved_docs
                ]

                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieved_chunks": len(retrieved_docs)
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                logger.error(f"Error in RAG query after {max_retries} attempts: {str(e)}")
                return {
                    "answer": f"An error occurred: {str(e)}",
                    "sources": [],
                    "retrieved_chunks": 0
                }

    def get_papers_summary(self) -> Dict:
        """Get overview of papers in the knowledge base"""
        return self.vector_store.get_papers_summary()

    def clear_knowledge_base(self) -> None:
        """Clear all papers from the knowledge base"""
        self.vector_store.clear()
        self._knowledge_base_version += 1
        self._cached_suggestions = None
        logger.info("Knowledge base cleared")

    def suggest_research_questions(self) -> List[str]:
        """Generate 1–3 context-relevant research questions based on loaded papers"""
        if self._cached_suggestions and self._knowledge_base_version == 0:
            return self._cached_suggestions
            
        papers_summary = self.get_papers_summary()
        if not papers_summary:
            return []
        
        try:
            # Create a summary of papers for the prompt
            summary_text = ""
            for source, info in papers_summary.items():
                summary_text += f"Paper: {info['title']}\nAuthors: {info['authors']}\nYear: {info['year']}\nSections: {', '.join(info['sections'])}\n---\n"
            
            # Generate prompt for questions
            prompt = f"""You are an academic research assistant. Based on the provided summary of research papers, generate 1–3 specific, context-relevant research questions to guide the user. Focus on methodologies, gaps, or comparisons. Use concise, academic language. Return only the list of questions, each starting with a dash (-).

PAPER SUMMARY:
{summary_text}

Generate 1–3 research questions based on the above papers."""
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.5,
                    top_p=0.9
                )
            )
            
            # Parse response to extract questions
            questions = [q.strip() for q in response.text.strip().split('\n') if q.strip() and q.strip().startswith('-')]
            questions = [q.lstrip('- ').strip() for q in questions[:3]]  # Limit to 3 questions
            self._cached_suggestions = questions
            self._knowledge_base_version = 0
            logger.info(f"Generated {len(questions)} context-relevant questions")
            return questions
        
        except Exception as e:
            logger.error(f"Error generating research questions: {str(e)}")
            return []
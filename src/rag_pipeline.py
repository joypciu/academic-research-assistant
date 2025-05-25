from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import logging
from .vector_store import FAISSVectorStore
from .utils import load_sample_questions

logger = logging.getLogger(__name__)

class ResearchRAGPipeline:
    # Define prompt templates for generating section-specific research questions
    PROMPT_TEMPLATES = {
        "Abstract": "Based on the following abstracts from multiple research papers, suggest a research question that captures the main focus of these studies.",
        "Methodology": "Based on the following methodology descriptions from multiple research papers, suggest a research question that compares or analyzes the different approaches used.",
        "Results": "Based on the following results from multiple research papers, suggest a research question that discusses or compares the findings.",
        "Discussion": "Based on the following discussion sections from multiple research papers, suggest a research question that explores the implications or interpretations of the results."
    }

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.vector_store = FAISSVectorStore()
        self._cached_suggestions = None
        self._knowledge_base_version = 0  # Track changes to knowledge base
        self._cached_version = -1  # Track when suggestions were last generated

    def add_papers(self, documents: List) -> None:
        """Add research papers to the knowledge base"""
        self.vector_store.add_documents(documents)
        self._knowledge_base_version += 1  # Invalidate suggestions cache
        self._cached_suggestions = None

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
Content: {doc.page_content[:500]}...
Relevance Score: {score:.3f}
---""")
        
        return "\n".join(context_parts)

    def generate_research_prompt(self, query: str, context: str) -> str:
        """Generate specialized prompt for research queries"""
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

    def query(self, question: str, k: int = 5, filter_section: str = None) -> Dict:
        """Query the research paper knowledge base with retry logic"""
        import time
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                retrieved_docs = self.vector_store.similarity_search(
                    question, k=k, filter_section=filter_section
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
                    max_output_tokens=800,
                    temperature=0.3,
                    top_p=0.8
                )
                
                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=config
                )
                
                sources = [
                    {
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'section': doc.metadata.get('section', 'Unknown'),
                        'authors': doc.metadata.get('authors', 'Unknown Authors'),
                        'year': doc.metadata.get('year', 'Unknown'),
                        'relevance_score': round(score, 3),
                        'content_preview': doc.page_content[:200] + "..."
                    } for doc, score in retrieved_docs
                ]
                
                return {
                    "answer": response.text,
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
                    "answer": f"Error processing query: {str(e)}",
                    "sources": [],
                    "retrieved_chunks": 0
                }

    def get_papers_overview(self) -> Dict:
        """Get overview of papers in the knowledge base"""
        return self.vector_store.get_papers_summary()

    def clear_knowledge_base(self) -> None:
        """Clear all papers from the knowledge base"""
        self.vector_store.clear()
        self._knowledge_base_version += 1
        self._cached_suggestions = None

    def suggest_research_questions(self) -> List[str]:
        """Generate dynamic research questions based on uploaded PDFs"""
        # Check if cached suggestions are valid
        # if self._cached_suggestions is not None and self._cached_version == self._knowledge_base_version:
        #     return self._cached_suggestions

        suggested_questions = []

        # Generate questions for each key section
        for section, template in self.PROMPT_TEMPLATES.items():
            snippets = []
            for source in set(doc.metadata['source'] for doc in self.vector_store.documents):
                section_chunks = [
                    doc for doc in self.vector_store.documents
                    if doc.metadata['source'] == source and doc.metadata['section'] == section
                ]
                if section_chunks:
                    snippet = section_chunks[0].page_content[:200]
                    snippets.append(f"Paper: {source}\n{snippet}\n")
                    logger.info(f"Collected snippet for section {section} from {source}")
                else:
                    logger.info(f"No snippets found for section {section} in {source}")

            if snippets:
                context = "\n".join(snippets)
                prompt = f"You are an AI research assistant. {template} Provide only the research question.\n\nExcerpts:\n{context}\n\nResearch Question:"
                logger.info(f"Generating question for {section} with prompt: {prompt[:100]}...")
                
                try:
                    config = types.GenerateContentConfig(
                        max_output_tokens=100,
                        temperature=0.7,
                        top_p=0.8
                    )
                    response = self.client.models.generate_content(
                        model="gemini-3-12b-it",
                        contents=prompt,
                        config=config
                    )
                    question = response.text.strip()
                    if question:
                        suggested_questions.append(question)
                        logger.info(f"Generated question for {section}: {question}")
                except Exception as e:
                    logger.error(f"Error generating question for section {section}: {e}")

        # Cache the new suggestions
        self._cached_suggestions = suggested_questions
        self._cached_version = self._knowledge_base_version

        return suggested_questions
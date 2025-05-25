from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import logging
import random

logger = logging.getLogger(__name__)

class ResearchRAGPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.vector_store = FAISSVectorStore()
        self._cached_suggestions = None
        self._knowledge_base_version = 0  # Track changes to knowledge base
    
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
        """Generate relevant research questions dynamically based on uploaded papers"""
        if self._cached_suggestions is not None and self._knowledge_base_version == 0:
            return self._cached_suggestions
        
        papers_summary = self.get_papers_overview()
        if not papers_summary:
            self._cached_suggestions = []
            return self._cached_suggestions
        
        # Create a summary of papers for the prompt
        paper_summaries = []
        for filename, info in papers_summary.items():
            paper_summary = f"Title: {info['title']}\nAuthors: {info['authors']}\nYear: {info['year']}\nSections: {', '.join(info['sections'])}\n"
            # Sample a few document chunks to get content insights (e.g., from Abstract or Introduction)
            docs = [doc for doc in self.vector_store.documents if doc.metadata.get('source') == filename]
            if docs:
                abstract_docs = [doc for doc in docs if doc.metadata.get('section') == 'Abstract'][:2]
                if abstract_docs:
                    paper_summary += "Abstract Excerpt: " + " ".join([doc.page_content[:200] for doc in abstract_docs]) + "\n"
            paper_summaries.append(paper_summary)
        
        context = "\n".join(paper_summaries) if paper_summaries else "No papers available."
        
        # Prompt for generating research questions
        prompt = f"""You are an AI research assistant tasked with generating insightful research questions based on a collection of academic papers.
INSTRUCTIONS:
- Generate 5-10 specific, relevant research questions based on the provided paper summaries.
- Focus on questions that encourage comparison, analysis of methodologies, identification of research gaps, evaluation of results, or exploration of datasets and contributions.
- Ensure questions are concise, academically rigorous, and tailored to the provided papers.
- Avoid generic questions; make them specific to the papers' content or metadata (e.g., titles, authors, years, sections).
- Format the output as a numbered list.

PAPER SUMMARIES:
{context}

OUTPUT:
Generate a numbered list of 5-10 research questions.
"""
        
        try:
            config = types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.5,  # Slightly higher for creative question generation
                top_p=0.9
            )
            
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=config
            )
            
            # Parse the response into a list of questions
            questions = []
            for line in response.text.split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and "." in line:
                    question = line[line.find(".") + 1:].strip()
                    if question:
                        questions.append(question)
            
            self._cached_suggestions = questions[:10]  # Limit to 10 questions
            self._knowledge_base_version = 0  # Reset to allow caching until new papers are added
            logger.info(f"Generated {len(self._cached_suggestions)} research questions")
            return self._cached_suggestions
        
        except Exception as e:
            logger.error(f"Error generating research questions: {str(e)}")
            self._cached_suggestions = [
                "What methodologies are used across these papers?",
                "Compare the evaluation metrics used in these studies",
                "What datasets were used for experiments?",
                "What are the main contributions of each paper?",
                "Identify research gaps mentioned in the papers"
            ]  # Fallback to generic questions
            return self._cached_suggestions
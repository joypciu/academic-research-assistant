import pypdf
import re
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class AcademicPaperProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_metadata_from_text(self, text: str) -> Dict:
        """Extract paper metadata like title, authors, year"""
        metadata = {}
        
        # Extract title (usually first few lines)
        lines = text.split('\n')[:10]
        potential_title = max(lines, key=len) if lines else "Unknown Title"
        metadata['title'] = potential_title[:100]
        
        # Extract year
        year_match = re.search(r'(19|20)\d{2}', text[:2000])
        metadata['year'] = year_match.group() if year_match else "Unknown"
        
        # Extract authors (simple heuristic)
        author_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z]\.\s*[A-Z][a-z]+)'
        ]
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:1000])
            authors.extend(matches[:3])  # Take first 3 matches
        metadata['authors'] = ', '.join(authors[:3]) if authors else "Unknown Authors"
        
        return metadata
    
    def identify_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify paper sections and their positions"""
        sections = []
        section_patterns = [
            (r'(?i)\b(abstract)\b', 'Abstract'),
            (r'(?i)\b(introduction)\b', 'Introduction'),
            (r'(?i)\b(methodology|methods?)\b', 'Methodology'),
            (r'(?i)\b(results?)\b', 'Results'),
            (r'(?i)\b(discussion)\b', 'Discussion'),
            (r'(?i)\b(conclusion)\b', 'Conclusion'),
            (r'(?i)\b(references?)\b', 'References'),
            (r'(?i)\b(related\s+work)\b', 'Related Work'),
            (r'(?i)\b(experiments?)\b', 'Experiments'),
            (r'(?i)\b(evaluation)\b', 'Evaluation')
        ]
        
        for pattern, section_name in section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                sections.append((section_name, match.start(), match.end()))
        
        # Sort by position
        sections.sort(key=lambda x: x[1])
        return sections
    
    def process_pdf(self, pdf_file, filename: str) -> List[Document]:
        """Process PDF and return structured documents with metadata"""
        try:
            # pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                return []
            
            # Extract metadata
            paper_metadata = self.extract_metadata_from_text(text)
            paper_metadata['source'] = filename
            paper_metadata['total_pages'] = len(pdf_reader.pages)
            
            # Identify sections
            sections = self.identify_sections(text)
            
            # Create chunks with section awareness
            documents = []
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # Determine which section this chunk belongs to
                chunk_start = text.find(chunk)
                current_section = "Unknown"
                
                for section_name, start, end in sections:
                    if start <= chunk_start:
                        current_section = section_name
                
                chunk_metadata = paper_metadata.copy()
                chunk_metadata.update({
                    'section': current_section,
                    'chunk_id': i,
                    'chunk_start': chunk_start
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.info(f"Processed {filename}: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            return []
import pypdf
import re
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class AcademicPaperProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for metadata extraction")
        except:
            logger.warning("spaCy model not found. Falling back to regex-based metadata extraction. Install with: pip install spacy && python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_metadata_from_text(self, text: str) -> Dict:
        """Extract paper metadata like title, authors, year using spaCy or regex"""
        metadata = {}
        
        # Extract title (first non-empty line in first 10 lines)
        lines = [line.strip() for line in text.split('\n')[:10] if line.strip()]
        metadata['title'] = max(lines, key=len, default="Unknown Title")[:100]
        
        # Extract year
        year_match = re.search(r'(19|20)\d{2}', text[:2000])
        metadata['year'] = year_match.group() if year_match else "Unknown"
        
        # Extract authors
        authors = []
        if self.nlp:
            doc = self.nlp(text[:1000])
            authors = [ent.text for ent in doc.ents if ent.label_ == "PERSON"][:3]
        else:
            author_patterns = [
                r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z]\.\s*[A-Z][a-z]+)'
            ]
            for pattern in author_patterns:
                matches = re.findall(pattern, text[:1000])
                authors.extend(matches[:3])
        metadata['authors'] = ', '.join(authors[:3]) if authors else "Unknown Authors"
        
        logger.info(f"Extracted metadata: title={metadata['title'][:50]}..., year={metadata['year']}, authors={metadata['authors']}")
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
        
        sections.sort(key=lambda x: x[1])
        logger.info(f"Identified {len(sections)} sections: {[s[0] for s in sections]}")
        return sections
    
    def process_pdf(self, pdf_file, filename: str) -> List[Document]:
        """Process a single PDF and return structured documents with metadata"""
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                return []
            
            paper_metadata = self.extract_metadata_from_text(text)
            paper_metadata['source'] = filename
            paper_metadata['total_pages'] = len(pdf_reader.pages)
            
            sections = self.identify_sections(text)
            documents = []
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_start = text.find(chunk)
                current_section = "Unknown"
                
                for section_name, start, _ in sections:
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
    
    def process_pdfs_parallel(self, pdf_files: List, filenames: List[str]) -> List[Document]:
        """Process multiple PDFs in parallel"""
        documents = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(self.process_pdf, pdf_file, filename): filename 
                            for pdf_file, filename in zip(pdf_files, filenames)}
            for future in as_completed(future_to_file):
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing file {future_to_file[future]}: {str(e)}")
        logger.info(f"Processed {len(documents)} total documents from {len(filenames)} files")
        return documents
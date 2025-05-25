import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Advanced FAISS-based vector store for academic research papers.
    Supports multiple embedding backends with fallback options.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 fallback_to_tfidf: bool = True):
        self.embedding_model_name = embedding_model
        self.fallback_to_tfidf = fallback_to_tfidf
        self.embedding_backend = None
        self.dimension = None
        self.index = None
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None
        self.document_metadata: List[Dict] = []
        
        # Initialize embedding backend
        self._initialize_embedding_backend()
        
        # Initialize FAISS index
        if self.dimension:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
    
    def _initialize_embedding_backend(self):
        """Initialize embedding backend with fallback options"""
        
        # Try SentenceTransformers first
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_backend = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_backend.get_sentence_embedding_dimension()
            self.backend_type = "sentence_transformers"
            logger.info(f"Using SentenceTransformers backend: {self.embedding_model_name}")
            return
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformers: {e}")
            
        # Try OpenAI embeddings if API key available
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.embedding_backend = "openai"
                self.dimension = 1536  # text-embedding-ada-002 dimension
                self.backend_type = "openai"
                logger.info("Using OpenAI embeddings backend")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            
        # Try Hugging Face transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.dimension = 384
            self.backend_type = "huggingface"
            logger.info("Using Hugging Face transformers backend")
            return
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face transformers: {e}")
            
        # Fallback to TF-IDF
        if self.fallback_to_tfidf:
            self.embedding_backend = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                sublinear_tf=True,
                use_idf=True
            )
            self.dimension = 1000
            self.backend_type = "tfidf"
            self.tfidf_fitted = False
            logger.info("Using TF-IDF fallback backend")
        else:
            raise RuntimeError("No embedding backend available and fallback disabled")
    
    def _encode_texts_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode texts using SentenceTransformers"""
        embeddings = self.embedding_backend.encode(
            texts, 
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        return embeddings.astype('float32')
    
    def _encode_texts_openai(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API"""
        import openai
        
        embeddings = []
        batch_size = 100  # OpenAI rate limit consideration
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                # Fallback to zeros for failed batches
                embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return np.array(embeddings, dtype='float32')
    
    def _encode_texts_huggingface(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Hugging Face transformers"""
        import torch
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            embeddings.extend(batch_embeddings.numpy())
        
        return np.array(embeddings, dtype='float32')
    
    def _encode_texts_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        if not self.tfidf_fitted:
            # Fit on first batch
            embeddings = self.embedding_backend.fit_transform(texts)
            self.tfidf_fitted = True
        else:
            embeddings = self.embedding_backend.transform(texts)
        
        # Convert sparse to dense
        embeddings = embeddings.toarray().astype('float32')
        
        # Ensure consistent dimensions
        if embeddings.shape[1] < self.dimension:
            padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        elif embeddings.shape[1] > self.dimension:
            embeddings = embeddings[:, :self.dimension]
            
        return embeddings
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the selected backend"""
        if self.backend_type == "sentence_transformers":
            return self._encode_texts_sentence_transformers(texts)
        elif self.backend_type == "openai":
            return self._encode_texts_openai(texts)
        elif self.backend_type == "huggingface":
            return self._encode_texts_huggingface(texts)
        elif self.backend_type == "tfidf":
            return self._encode_texts_tfidf(texts)
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store with enhanced metadata handling"""
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract text for embedding
        texts = [doc.page_content for doc in documents]
        
        try:
            # Generate embeddings
            embeddings = self.encode_texts(texts)
            
            # Normalize for cosine similarity (except for TF-IDF which is already normalized)
            if self.backend_type != "tfidf":
                faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            self.document_metadata.extend([doc.metadata for doc in documents])
            
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
            
            logger.info(f"Successfully added {len(documents)} documents. "
                       f"Vector store now contains {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_section: str = None,
                         filter_year: str = None,
                         min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Enhanced similarity search with multiple filtering options
        """
        if len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.encode_texts([query])
            
            if self.backend_type != "tfidf":
                faiss.normalize_L2(query_embedding)
            
            # Search with larger k to allow for filtering
            search_k = min(k * 3, len(self.documents))
            scores, indices = self.index.search(query_embedding, search_k)
            
            results = []
            seen_content = set()  # Avoid near-duplicates
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Apply filters
                    if filter_section:
                        doc_section = doc.metadata.get('section', '').lower()
                        if filter_section.lower() not in doc_section:
                            continue
                    
                    if filter_year:
                        doc_year = str(doc.metadata.get('year', ''))
                        if filter_year not in doc_year:
                            continue
                    
                    # Score threshold
                    if float(score) < min_score:
                        continue
                    
                    # Avoid near-duplicate content
                    content_hash = hash(doc.page_content[:200])
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)
                    
                    results.append((doc, float(score)))
                    
                    if len(results) >= k:
                        break
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_papers_summary(self) -> Dict:
        """Get comprehensive summary of papers in the vector store"""
        if not self.documents:
            return {}
        
        papers = {}
        section_stats = {}
        year_stats = {}
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            year = doc.metadata.get('year', 'Unknown')
            
            # Paper-level stats
            if source not in papers:
                papers[source] = {
                    'title': doc.metadata.get('title', 'Unknown Title'),
                    'authors': doc.metadata.get('authors', 'Unknown Authors'),
                    'year': year,
                    'sections': set(),
                    'chunk_count': 0,
                    'total_length': 0
                }
            
            papers[source]['sections'].add(section)
            papers[source]['chunk_count'] += 1
            papers[source]['total_length'] += len(doc.page_content)
            
            # Global stats
            section_stats[section] = section_stats.get(section, 0) + 1
            year_stats[year] = year_stats.get(year, 0) + 1
        
        # Convert sets to lists for JSON serialization
        for paper in papers.values():
            paper['sections'] = sorted(list(paper['sections']))
        
        # Return just the papers dict directly to match the expected structure
        return papers
    
    def get_similar_papers(self, paper_title: str, k: int = 3) -> List[Dict]:
        """Find papers similar to a given paper"""
        target_docs = [doc for doc in self.documents 
                      if paper_title.lower() in doc.metadata.get('title', '').lower()]
        
        if not target_docs:
            return []
        
        # Use first chunk of target paper as query
        target_content = target_docs[0].page_content
        similar_docs = self.similarity_search(target_content, k=k*2)
        
        # Group by paper and exclude the target paper
        similar_papers = {}
        for doc, score in similar_docs:
            source = doc.metadata.get('source', 'Unknown')
            title = doc.metadata.get('title', 'Unknown')
            
            if title.lower() != paper_title.lower():
                if source not in similar_papers:
                    similar_papers[source] = {
                        'title': title,
                        'authors': doc.metadata.get('authors', 'Unknown'),
                        'year': doc.metadata.get('year', 'Unknown'),
                        'max_similarity': score,
                        'matching_sections': []
                    }
                
                similar_papers[source]['matching_sections'].append({
                    'section': doc.metadata.get('section', 'Unknown'),
                    'similarity': score
                })
                
                if score > similar_papers[source]['max_similarity']:
                    similar_papers[source]['max_similarity'] = score
        
        # Sort by similarity and return top k
        sorted_papers = sorted(similar_papers.values(), 
                             key=lambda x: x['max_similarity'], 
                             reverse=True)
        
        return sorted_papers[:k]
    
    def save_to_disk(self, filepath: str):
        """Save vector store to disk"""
        try:
            save_data = {
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'embedding_model_name': self.embedding_model_name,
                'backend_type': self.backend_type,
                'dimension': self.dimension,
                'papers_summary': self.get_papers_summary()
            }
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save document data
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save embeddings if available
            if self.document_embeddings is not None:
                np.save(f"{filepath}_embeddings.npy", self.document_embeddings)
            
            logger.info(f"Vector store saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_from_disk(self, filepath: str):
        """Load vector store from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load document data
            with open(f"{filepath}.pkl", 'rb') as f:
                save_data = pickle.load(f)
            
            self.documents = save_data['documents']
            self.document_metadata = save_data['document_metadata']
            self.dimension = save_data['dimension']
            
            # Load embeddings if available
            embeddings_file = f"{filepath}_embeddings.npy"
            if os.path.exists(embeddings_file):
                self.document_embeddings = np.load(embeddings_file)
            
            # Reinitialize embedding backend
            self._initialize_embedding_backend()
            
            logger.info(f"Vector store loaded from {filepath}")
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def clear(self):
        """Clear the vector store"""
        if self.dimension:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        
        # Reset TF-IDF if using that backend
        if self.backend_type == "tfidf":
            self.tfidf_fitted = False
            
        logger.info("Vector store cleared")
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about the vector store"""
        if not self.documents:
            return {}
        
        summary = self.get_papers_summary()
        
        # Calculate additional statistics
        chunk_lengths = [len(doc.page_content) for doc in self.documents]
        
        stats = {
            'basic_info': summary['global_stats'],
            'content_statistics': {
                'avg_chunk_length': np.mean(chunk_lengths),
                'median_chunk_length': np.median(chunk_lengths),
                'min_chunk_length': np.min(chunk_lengths),
                'max_chunk_length': np.max(chunk_lengths),
                'total_characters': sum(chunk_lengths)
            },
            'paper_details': summary['papers']
        }
        
        return stats
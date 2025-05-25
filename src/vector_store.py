import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 fallback_to_tfidf: bool = True):
        self.embedding_model_name = embedding_model
        self.fallback_to_tfidf = fallback_to_tfidf
        self.embedding_backend = None
        self.backend_type = None
        self.dimension = None
        self.index = None
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None
        self.document_metadata: List[Dict] = []
        self.db_path = "papers.db"
        
        # Initialize SQLite database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for storing documents and embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source TEXT,
                title TEXT,
                authors TEXT,
                year TEXT,
                section TEXT,
                chunk_id INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
        """)
        conn.commit()
        conn.close()
    
    def _initialize_embedding_backend(self):
        """Initialize embedding backend with fallback options"""
        if self.embedding_backend is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_backend = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_backend.get_sentence_embedding_dimension()
            self.backend_type = "sentence_transformers"
            logger.info(f"Using SentenceTransformers: {self.embedding_model_name}")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            return
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformers: {e}")
        
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.embedding_backend = "openai"
                self.dimension = 1536
                self.backend_type = "openai"
                logger.info("Using OpenAI embeddings")
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 40
                return
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.dimension = 384
            self.backend_type = "huggingface"
            logger.info("Using Hugging Face transformers")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            return
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face transformers: {e}")
        
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
            logger.info("Using TF-IDF fallback")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
        else:
            raise RuntimeError("No embedding backend available")
    
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
        batch_size = 100
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
                embeddings.extend([[0.0] * self.dimension] * len(batch))
        return np.array(embeddings, dtype='float32')
    
    def _encode_texts_huggingface(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Hugging Face transformers"""
        import torch
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings.extend(batch_embeddings.numpy())
        return np.array(embeddings, dtype='float32')
    
    def _encode_texts_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        if not self.tfidf_fitted:
            embeddings = self.embedding_backend.fit_transform(texts)
            self.tfidf_fitted = True
        else:
            embeddings = self.embedding_backend.transform(texts)
        embeddings = embeddings.toarray().astype('float32')
        if embeddings.shape[1] < self.dimension:
            padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        elif embeddings.shape[1] > self.dimension:
            embeddings = embeddings[:, :self.dimension]
        return embeddings
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the selected backend"""
        if self.embedding_backend is None:
            self._initialize_embedding_backend()
        if self.backend_type == "sentence_transformers":
            return self._encode_texts_sentence_transformers(texts)
        elif self.backend_type == "openai":
            return self._encode_texts_openai(texts)
        elif self.backend_type == "huggingface":
            return self._encode_texts_huggingface(texts)
        elif self.backend_type == "tfidf":
            return self._encode_texts_tfidf(texts)
        raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store and SQLite database"""
        if not documents:
            logger.warning("No documents provided")
            return
        
        logger.info(f"Adding {len(documents)} documents")
        texts = [doc.page_content for doc in documents]
        try:
            embeddings = self.encode_texts(texts)
            if self.backend_type != "tfidf":
                faiss.normalize_L2(embeddings)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for doc, embedding in zip(documents, embeddings):
                cursor.execute("""
                    INSERT INTO documents (source, title, authors, year, section, chunk_id, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.metadata.get('source', 'Unknown'),
                    doc.metadata.get('title', 'Unknown Title'),
                    doc.metadata.get('authors', 'Unknown Authors'),
                    doc.metadata.get('year', 'Unknown'),
                    doc.metadata.get('section', 'Unknown'),
                    doc.metadata.get('chunk_id', 0),
                    doc.page_content
                ))
                doc_id = cursor.lastrowid
                cursor.execute("INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)",
                            (doc_id, pickle.dumps(embedding)))
            
            conn.commit()
            conn.close()
            
            self.index.add(embeddings)
            self.documents.extend(documents)
            self.document_metadata.extend([doc.metadata for doc in documents])
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
            
            logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_section: str = None,
                         filter_year: str = None,
                         min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """Enhanced similarity search with filtering"""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        try:
            query_embedding = self.encode_texts([query])
            if self.backend_type != "tfidf":
                faiss.normalize_L2(query_embedding)
            
            search_k = min(k * 3, len(self.documents))
            scores, indices = self.index.search(query_embedding, search_k)
            
            results = []
            seen_content = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    if filter_section and filter_section.lower() not in doc.metadata.get('section', '').lower():
                        continue
                    if filter_year and filter_year not in str(doc.metadata.get('year', '')):
                        continue
                    if float(score) < min_score:
                        continue
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
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in papers:
                papers[source] = {
                    'title': doc.metadata.get('title', 'Unknown Title'),
                    'authors': doc.metadata.get('authors', 'Unknown Authors'),
                    'year': doc.metadata.get('year', 'Unknown'),
                    'sections': set(),
                    'chunk_count': 0,
                    'total_length': 0
                }
            papers[source]['sections'].add(doc.metadata.get('section', 'Unknown'))
            papers[source]['chunk_count'] += 1
            papers[source]['total_length'] += len(doc.page_content)
        
        for paper in papers.values():
            paper['sections'] = sorted(list(paper['sections']))
        
        return papers
    
    def clear(self) -> None:
        """Clear the vector store and database"""
        if self.dimension:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        if self.backend_type == "tfidf":
            self.tfidf_fitted = False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        
        logger.info("Vector store and database cleared")
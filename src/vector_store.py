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
    def __init__(self, embedding_model: str = "tfidf"):
        self.embedding_model_name = embedding_model
        self.embedding_backend = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            use_idf=True
        )
        self.backend_type = "tfidf"
        self.dimension = 1000
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None
        self.document_metadata: List[Dict] = []
        self.db_path = "papers.db"
        self.tfidf_fitted = False
        
        # Initialize SQLite database
        self._init_db()
        logger.info("Initialized FAISSVectorStore with TF-IDF backend")

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
        logger.info("SQLite database initialized")

    def _encode_texts_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        if not texts:
            logger.warning("No texts provided for encoding")
            return np.zeros((0, self.dimension), dtype='float32')
        try:
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
            logger.info(f"Encoded {len(texts)} texts with TF-IDF, shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts with TF-IDF: {str(e)}")
            raise

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        return self._encode_texts_tfidf(texts)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store and SQLite database"""
        if not documents:
            logger.warning("No documents provided")
            return

        logger.info(f"Adding {len(documents)} documents")
        texts = [doc.page_content for doc in documents]
        try:
            embeddings = self.encode_texts(texts)
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
            logger.error(f"Error adding documents: {str(e)}")
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

            logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
            if not results:
                logger.warning("No documents matched query or filters")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
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
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.documents = []
        self.document_metadata = []
        self.document_embeddings = None
        self.tfidf_fitted = False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

        logger.info("Vector store and database cleared")
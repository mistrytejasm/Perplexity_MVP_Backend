import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import time
import os
import pickle

logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self):
        # initialize chromadb client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db") 

        # initialize embedding model
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")

        # get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Document Chunks for semantic search"}
        )

        # session tracking (persistance)
        self.session_file = "chroma_db/session_documents.pkl"
        self.session_documents = self._load_session()

        logger.info("DocumentStore Initialized")

    def _load_session(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load session documents from file"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'rb') as f:
                    sessions = pickle.load(f)
                logger.info(f"Loaded {len(sessions)} sessions")
                return sessions
        except Exception as e:
            logger.info(f"Could not load sessions: {e}")
        
        return {}
    
    def _save_sessions(self):
        """Save session documents to file"""
        try:
            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.session_documents, f)
            logger.info(f"Saved {len(self.session_documents)} sessions")
        except Exception as e:
            logger.error(f"Could not save sessions: {e}")


    def store_document_chunks(self, chunks: List[Any], document_metadata: Dict[str, Any]):
        """Store documents chunks with embeddings in ChromaDB"""

        logger.info(f"Storing len{(chunks)} chunks")

        # step 1: prepare data, text chunks
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # step 2: Generating Embedding
        start_time = time.time()

        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        logger.info(f"Generating embeddings in {time.time() - start_time:.2f}s")

        # prepare metadata
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "filename": document_metadata.get("filename", "unknown"),
                "session_id": document_metadata.get("session_id", "")
            }
            for chunk in chunks
        ]

        # step 3: Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=chunk_ids
        )

        logger.info(f"Store {len(chunks)} chunks")
        return len(chunks)
    
    def search_documents(self, query: str, session_id: str, max_results: int=5)-> List[Dict[str, Any]]:
        """Search for relevant documents in the session"""
        logger.info(f"Searching: '{query[:50]}...'")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()

            # search Chromadb
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=max_results,
                where={"session_id": session_id},
                include = ["documents", "metadatas", "distances"]
            )


            if not results["documents"][0]:
                logger.info(f"No Document found for session {session_id}")
                return []
            
            # convert to standardized format
            doc_results = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            ):
                similarity_score = 1 - distance
                doc_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                })

            logger.info(f"Found {len(doc_results)} results")
            return doc_results
        
        except Exception as e:
            logger.error(f"Search Error: {e}")
            return []
    
    def add_document_to_session(
        self, 
        session_id: str, 
        document_id: str, 
        document_info: Dict[str, Any]
    ):
        """Track document for a session"""
        if session_id not in self.session_documents:
            self.session_documents[session_id] = []
        
        document_info["document_id"] = document_id
        self.session_documents[session_id].append(document_info)
        self._save_sessions()
        
        logger.info(f"Added document to session {session_id}")
    
    def get_session_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a session"""
        return self.session_documents.get(session_id, [])
    
    def has_documents(self, session_id: str) -> bool:
        """Check if session has any documents"""
        return (
            session_id in self.session_documents and 
            len(self.session_documents[session_id]) > 0
        )
    
    def get_relevant_context(
        self, 
        query: str, 
        session_id: str, 
        max_chunks: int = 3,
        min_similarity: float = 0.3
    ) -> str:
        """Get relevant context for RAG"""
        
        # Check if session has documents
        if not self.has_documents(session_id):
            return ""
        
        # Search for relevant chunks
        results = self.search_documents(query, session_id, max_results=max_chunks)
        
        # Filter by minimum similarity
        relevant_results = [
            r for r in results 
            if r['similarity_score'] >= min_similarity
        ]
        
        if not relevant_results:
            logger.info(f"No relevant chunks above threshold {min_similarity}")
            return ""
        
        # Combine context
        context_parts = []
        for i, result in enumerate(relevant_results, 1):
            context_parts.append(
                f"[Source {i}] {result['content']}"
            )
        
        context = "\n\n".join(context_parts)
        logger.info(f"Built context from {len(relevant_results)} chunks")
        
        return context
    
    def clear_session(self, session_id: str):
        """Clear all documents from a session"""
        if session_id in self.session_documents:
            del self.session_documents[session_id]
            self._save_sessions()
            logger.info(f"Cleared session {session_id}")

            







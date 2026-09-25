
import logging
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import tool

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
)
from .models import ChunkResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

logger = logging.getLogger(__name__)

# Initialize embedding client with flexible provider
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector
    """
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")
class DocumentInput(BaseModel):
    """Input for document retrieval."""
    document_id: str = Field(..., description="Document ID to retrieve")
class DocumentListInput(BaseModel):
    """Input for listing documents."""
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")

@tool(
    "vector_search",
    description="Search document chunks by semantic similarity and return the most relevant results.",
    args_schema=VectorSearchInput,
)
async def vector_search_tool(
    query: str,
    limit: int = 10,
) -> List[ChunkResult]:
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(query)
        
        # Perform vector search
        results = await vector_search(
            embedding=embedding,
            limit=limit
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

@tool(
    "hybrid_search",
    description="Combine semantic vector search and keyword search for comprehensive document results.",
    args_schema=HybridSearchInput,
)
async def hybrid_search_tool(
    query: str,
    limit: int = 10,
    text_weight: float = 0.3,
) -> List[ChunkResult]:
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(query)
        
        # Perform hybrid search
        results = await hybrid_search(
            embedding=embedding,
            query_text=query,
            limit=limit,
            text_weight=text_weight
        )
        
        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []

@tool(
    "get_document",
    description="Retrieve bounded metadata for a document by ID. Use vector or hybrid search to retrieve document content.",
    args_schema=DocumentInput,
)
async def get_document_tool(
    document_id: str,
) -> Optional[DocumentMetadata]:
    try:
        document = await get_document(document_id)

        if not document:
            return None

        return DocumentMetadata(
            id=str(document["id"]),
            title=document["title"],
            source=document["source"],
            created_at=datetime.fromisoformat(document["created_at"]),
            updated_at=datetime.fromisoformat(document["updated_at"]),
        )
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None

@tool(
    "list_documents",
    description="List ingested documents with their metadata and chunk counts.",
    args_schema=DocumentListInput,
)
async def list_documents_tool(
    limit: int = 20,
    offset: int = 0,
) -> List[DocumentMetadata]:
    try:
        documents = await list_documents(
            limit=limit,
            offset=offset
        )
        
        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count")
            )
            for d in documents
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []


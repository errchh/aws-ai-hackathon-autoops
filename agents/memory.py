"""Memory layer implementation using ChromaDB for agent decision storage and retrieval."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
import boto3
from botocore.exceptions import ClientError

from config.settings import get_settings
from models.core import AgentDecision, MarketEvent


class EmbeddingService:
    """Service for generating embeddings using AWS Bedrock."""
    
    def __init__(self):
        """Initialize the embedding service with AWS Bedrock client."""
        settings = get_settings()
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws.region,
            aws_access_key_id=settings.aws.access_key_id,
            aws_secret_access_key=settings.aws.secret_access_key,
            aws_session_token=settings.aws.session_token
        )
        self.embedding_model_id = "amazon.titan-embed-text-v1"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using AWS Bedrock.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            body = json.dumps({
                "inputText": text
            })
            
            response = self.bedrock_client.invoke_model(
                modelId=self.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
            
        except ClientError as e:
            raise Exception(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))
        return embeddings


class AgentMemory:
    """ChromaDB-based memory system for storing and retrieving agent decisions."""
    
    def __init__(self):
        """Initialize the agent memory system."""
        self.embedding_service = EmbeddingService()
        self._client = None
        self._collection = None
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            settings = get_settings()
            # Create ChromaDB client with persistence
            self._client = chromadb.PersistentClient(
                path=settings.chromadb.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for agent memory
            self._collection = self._client.get_or_create_collection(
                name=settings.chromadb.collection_name,
                metadata={"description": "Agent decision memory for retail optimization"}
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {e}")
    
    def store_decision(
        self,
        agent_id: str,
        decision: AgentDecision,
        context: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an agent decision with context in the memory system.
        
        Args:
            agent_id: ID of the agent making the decision
            decision: The agent decision object
            context: Additional context for the decision
            outcome: Optional outcome data (can be added later)
            
        Returns:
            Unique ID for the stored memory entry
        """
        memory_id = str(uuid.uuid4())
        
        # Create searchable text for embedding
        search_text = self._create_search_text(agent_id, decision, context)
        
        # Generate embedding
        embedding = self.embedding_service.generate_embedding(search_text)
        
        # Prepare metadata
        metadata = {
            "agent_id": agent_id,
            "decision_id": str(decision.id),
            "timestamp": decision.timestamp.isoformat(),
            "action_type": decision.action_type.value,
            "confidence_score": decision.confidence_score,
            "has_outcome": outcome is not None
        }
        
        # Prepare document content
        document_content = {
            "decision": decision.model_dump(mode='json'),
            "context": context,
            "outcome": outcome
        }
        
        # Store in ChromaDB
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[json.dumps(document_content)],
            metadatas=[metadata]
        )
        
        return memory_id
    
    def update_outcome(self, memory_id: str, outcome: Dict[str, Any]):
        """
        Update the outcome for a stored decision.
        
        Args:
            memory_id: ID of the memory entry to update
            outcome: Outcome data to add
        """
        # Get existing entry
        result = self._collection.get(ids=[memory_id])
        if not result['documents']:
            raise ValueError(f"Memory entry {memory_id} not found")
        
        # Update document with outcome
        document_content = json.loads(result['documents'][0])
        document_content['outcome'] = outcome
        
        # Update metadata
        metadata = result['metadatas'][0]
        metadata['has_outcome'] = True
        
        # Update in ChromaDB
        self._collection.update(
            ids=[memory_id],
            documents=[json.dumps(document_content)],
            metadatas=[metadata]
        )
    
    def retrieve_similar_decisions(
        self,
        agent_id: str,
        current_context: Dict[str, Any],
        action_type: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve similar past decisions for the given context.
        
        Args:
            agent_id: ID of the agent to search for
            current_context: Current decision context
            action_type: Optional filter by action type
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples containing (decision_data, similarity_score)
        """
        # Create search text from current context
        search_text = self._create_context_search_text(current_context)
        
        # Generate embedding for search
        query_embedding = self.embedding_service.generate_embedding(search_text)
        
        # Prepare where clause for filtering with proper ChromaDB syntax
        conditions = [{"agent_id": {"$eq": agent_id}}]
        if action_type:
            conditions.append({"action_type": {"$eq": action_type}})
        
        where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause
        )
        
        # Process results
        similar_decisions = []
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0],
            results['distances'][0], 
            results['metadatas'][0]
        )):
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            similarity_score = 1 - distance
            
            if similarity_score >= similarity_threshold:
                decision_data = json.loads(doc)
                decision_data['metadata'] = metadata
                decision_data['similarity_score'] = similarity_score
                similar_decisions.append((decision_data, similarity_score))
        
        return similar_decisions
    
    def get_agent_decision_history(
        self,
        agent_id: str,
        action_type: Optional[str] = None,
        limit: int = 50,
        include_outcomes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get decision history for a specific agent.
        
        Args:
            agent_id: ID of the agent
            action_type: Optional filter by action type
            limit: Maximum number of results
            include_outcomes: Whether to include only decisions with outcomes
            
        Returns:
            List of decision records
        """
        # Build where clause with proper ChromaDB syntax
        conditions = [{"agent_id": {"$eq": agent_id}}]
        
        if action_type:
            conditions.append({"action_type": {"$eq": action_type}})
        if include_outcomes:
            conditions.append({"has_outcome": {"$eq": True}})
        
        # Use $and operator for multiple conditions
        where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        results = self._collection.get(
            where=where_clause,
            limit=limit
        )
        
        decision_history = []
        for doc, metadata in zip(results['documents'], results['metadatas']):
            decision_data = json.loads(doc)
            decision_data['metadata'] = metadata
            decision_history.append(decision_data)
        
        # Sort by timestamp (most recent first)
        decision_history.sort(
            key=lambda x: x['decision']['timestamp'],
            reverse=True
        )
        
        return decision_history
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system-wide memory metrics.
        
        Returns:
            Dictionary containing memory system metrics
        """
        # Get collection info
        collection_count = self._collection.count()
        
        # Get agent-specific counts
        agent_counts = {}
        for agent_id in ["pricing_agent", "inventory_agent", "promotion_agent"]:
            results = self._collection.get(where={"agent_id": {"$eq": agent_id}})
            agent_counts[agent_id] = len(results['documents'])
        
        # Get decisions with outcomes
        results_with_outcomes = self._collection.get(where={"has_outcome": {"$eq": True}})
        decisions_with_outcomes = len(results_with_outcomes['documents'])
        
        settings = get_settings()
        return {
            "total_decisions": collection_count,
            "decisions_with_outcomes": decisions_with_outcomes,
            "agent_decision_counts": agent_counts,
            "collection_name": settings.chromadb.collection_name,
            "persist_directory": settings.chromadb.persist_directory
        }
    
    def _create_search_text(
        self,
        agent_id: str,
        decision: AgentDecision,
        context: Dict[str, Any]
    ) -> str:
        """Create searchable text representation of a decision."""
        context_str = " ".join([
            f"{k}: {v}" for k, v in context.items()
            if isinstance(v, (str, int, float, bool))
        ])
        
        return f"""
        Agent: {agent_id}
        Action: {decision.action_type}
        Rationale: {decision.rationale}
        Context: {context_str}
        Confidence: {decision.confidence_score}
        """.strip()
    
    def _create_context_search_text(self, context: Dict[str, Any]) -> str:
        """Create searchable text from context for similarity search."""
        context_str = " ".join([
            f"{k}: {v}" for k, v in context.items()
            if isinstance(v, (str, int, float, bool))
        ])
        return context_str
    
    def reset_memory(self):
        """Reset the memory system (for testing purposes)."""
        if self._collection:
            settings = get_settings()
            self._client.delete_collection(settings.chromadb.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=settings.chromadb.collection_name,
                metadata={"description": "Agent decision memory for retail optimization"}
            )


# Global memory instance
agent_memory = AgentMemory()
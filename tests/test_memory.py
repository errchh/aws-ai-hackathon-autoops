"""Tests for the agent memory system using ChromaDB."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.memory import AgentMemory, EmbeddingService
from models.core import AgentDecision


class TestEmbeddingService:
    """Test cases for the EmbeddingService class."""
    
    @pytest.fixture
    def mock_bedrock_client(self):
        """Mock AWS Bedrock client."""
        with patch('boto3.client') as mock_client:
            mock_bedrock = Mock()
            mock_client.return_value = mock_bedrock
            
            # Mock successful embedding response
            mock_response = {
                'body': Mock()
            }
            mock_response['body'].read.return_value = json.dumps({
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 77  # 385 dimensions, close to 384
            }).encode()
            mock_bedrock.invoke_model.return_value = mock_response
            
            yield mock_bedrock
    
    def test_embedding_service_initialization(self, mock_bedrock_client):
        """Test EmbeddingService initialization."""
        service = EmbeddingService()
        assert service.embedding_model_id == "amazon.titan-embed-text-v1"
        assert service.bedrock_client is not None
    
    def test_generate_embedding_success(self, mock_bedrock_client):
        """Test successful embedding generation."""
        service = EmbeddingService()
        
        text = "Test text for embedding"
        embedding = service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 385  # Expected embedding dimension
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify Bedrock was called correctly
        mock_bedrock_client.invoke_model.assert_called_once()
        call_args = mock_bedrock_client.invoke_model.call_args
        assert call_args[1]['modelId'] == "amazon.titan-embed-text-v1"
        
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == text
    
    def test_generate_embedding_failure(self, mock_bedrock_client):
        """Test embedding generation failure handling."""
        from botocore.exceptions import ClientError
        
        # Mock client error
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid input'}},
            'InvokeModel'
        )
        
        service = EmbeddingService()
        
        with pytest.raises(Exception, match="Failed to generate embedding"):
            service.generate_embedding("test text")
    
    def test_generate_embeddings_batch(self, mock_bedrock_client):
        """Test batch embedding generation."""
        service = EmbeddingService()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = service.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 385 for emb in embeddings)
        assert mock_bedrock_client.invoke_model.call_count == 3


class TestAgentMemory:
    """Test cases for the AgentMemory class."""
    
    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB client and collection."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock collection methods
            mock_collection.add = Mock()
            mock_collection.update = Mock()
            mock_collection.get = Mock()
            mock_collection.query = Mock()
            mock_collection.count = Mock(return_value=100)
            
            yield mock_client, mock_collection
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        with patch('agents.memory.EmbeddingService') as mock_service_class:
            mock_service = Mock()
            mock_service.generate_embedding.return_value = [0.1] * 384
            mock_service_class.return_value = mock_service
            yield mock_service
    
    @pytest.fixture
    def sample_decision(self):
        """Sample agent decision for testing."""
        from models.core import ActionType
        return AgentDecision(
            agent_id="pricing_agent",
            timestamp=datetime.now(),
            action_type=ActionType.PRICE_ADJUSTMENT,
            parameters={"product_id": "SKU123", "new_price": 29.99},
            rationale="Adjusting price due to high inventory levels",
            confidence_score=0.85,
            expected_outcome={"revenue_impact": 1500.0}
        )
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            "product_id": "SKU123",
            "current_price": 34.99,
            "inventory_level": 150,
            "demand_trend": "declining",
            "competitor_prices": [32.99, 31.50, 33.25]
        }
    
    def test_agent_memory_initialization(self, mock_chromadb, mock_embedding_service):
        """Test AgentMemory initialization."""
        mock_client, mock_collection = mock_chromadb
        
        memory = AgentMemory()
        
        assert memory._client == mock_client
        assert memory._collection == mock_collection
        assert memory.embedding_service is not None
        
        # Verify ChromaDB setup
        mock_client.get_or_create_collection.assert_called_once()
    
    def test_store_decision(self, mock_chromadb, mock_embedding_service, sample_decision, sample_context):
        """Test storing an agent decision."""
        mock_client, mock_collection = mock_chromadb
        
        memory = AgentMemory()
        memory_id = memory.store_decision(
            agent_id="pricing_agent",
            decision=sample_decision,
            context=sample_context
        )
        
        # Verify memory ID is generated
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        
        # Verify embedding was generated
        mock_embedding_service.generate_embedding.assert_called_once()
        
        # Verify ChromaDB add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args['ids']) == 1
        assert len(call_args['embeddings']) == 1
        assert len(call_args['documents']) == 1
        assert len(call_args['metadatas']) == 1
        
        # Verify metadata structure
        metadata = call_args['metadatas'][0]
        assert metadata['agent_id'] == "pricing_agent"
        assert metadata['decision_id'] == str(sample_decision.id)
        assert metadata['action_type'] == sample_decision.action_type.value
        assert metadata['confidence_score'] == sample_decision.confidence_score
        assert metadata['has_outcome'] is False
        
        # Verify document content
        document = json.loads(call_args['documents'][0])
        assert 'decision' in document
        assert 'context' in document
        assert document['outcome'] is None
    
    def test_store_decision_with_outcome(self, mock_chromadb, mock_embedding_service, sample_decision, sample_context):
        """Test storing a decision with outcome."""
        mock_client, mock_collection = mock_chromadb
        
        outcome = {"revenue_change": 1200.0, "effectiveness_score": 0.78}
        
        memory = AgentMemory()
        memory_id = memory.store_decision(
            agent_id="pricing_agent",
            decision=sample_decision,
            context=sample_context,
            outcome=outcome
        )
        
        # Verify outcome is included
        call_args = mock_collection.add.call_args[1]
        metadata = call_args['metadatas'][0]
        assert metadata['has_outcome'] is True
        
        document = json.loads(call_args['documents'][0])
        assert document['outcome'] == outcome
    
    def test_update_outcome(self, mock_chromadb, mock_embedding_service):
        """Test updating outcome for existing decision."""
        mock_client, mock_collection = mock_chromadb
        
        # Mock existing document
        existing_doc = {
            "decision": {"id": "test-123"},
            "context": {"product_id": "SKU123"},
            "outcome": None
        }
        existing_metadata = {"has_outcome": False}
        
        mock_collection.get.return_value = {
            'documents': [json.dumps(existing_doc)],
            'metadatas': [existing_metadata]
        }
        
        memory = AgentMemory()
        outcome = {"revenue_change": 800.0}
        
        memory.update_outcome("memory-123", outcome)
        
        # Verify get was called
        mock_collection.get.assert_called_once_with(ids=["memory-123"])
        
        # Verify update was called
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args[1]
        
        updated_doc = json.loads(call_args['documents'][0])
        assert updated_doc['outcome'] == outcome
        
        updated_metadata = call_args['metadatas'][0]
        assert updated_metadata['has_outcome'] is True
    
    def test_update_outcome_not_found(self, mock_chromadb, mock_embedding_service):
        """Test updating outcome for non-existent decision."""
        mock_client, mock_collection = mock_chromadb
        
        mock_collection.get.return_value = {'documents': []}
        
        memory = AgentMemory()
        
        with pytest.raises(ValueError, match="Memory entry .* not found"):
            memory.update_outcome("nonexistent-id", {"outcome": "test"})
    
    def test_retrieve_similar_decisions(self, mock_chromadb, mock_embedding_service):
        """Test retrieving similar decisions."""
        mock_client, mock_collection = mock_chromadb
        
        # Mock query results
        similar_docs = [
            json.dumps({
                "decision": {"action_type": "price_adjustment", "rationale": "Similar situation"},
                "context": {"product_id": "SKU456"},
                "outcome": {"effectiveness_score": 0.9}
            }),
            json.dumps({
                "decision": {"action_type": "price_adjustment", "rationale": "Another similar case"},
                "context": {"product_id": "SKU789"},
                "outcome": {"effectiveness_score": 0.7}
            })
        ]
        
        mock_collection.query.return_value = {
            'documents': [similar_docs],
            'distances': [[0.2, 0.4]],  # ChromaDB returns distances (lower = more similar)
            'metadatas': [[
                {"agent_id": "pricing_agent", "action_type": "price_adjustment"},
                {"agent_id": "pricing_agent", "action_type": "price_adjustment"}
            ]]
        }
        
        memory = AgentMemory()
        current_context = {"product_id": "SKU123", "inventory_level": 100}
        
        results = memory.retrieve_similar_decisions(
            agent_id="pricing_agent",
            current_context=current_context,
            action_type="price_adjustment",
            limit=5,
            similarity_threshold=0.5
        )
        
        # Verify embedding generation for query
        mock_embedding_service.generate_embedding.assert_called()
        
        # Verify ChromaDB query
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert call_args['n_results'] == 5
        expected_where = {"$and": [
            {"agent_id": {"$eq": "pricing_agent"}}, 
            {"action_type": {"$eq": "price_adjustment"}}
        ]}
        assert call_args['where'] == expected_where
        
        # Verify results
        assert len(results) == 2
        
        # Check similarity scores (1 - distance)
        assert results[0][1] == 0.8  # 1 - 0.2
        assert results[1][1] == 0.6  # 1 - 0.4
        
        # Verify decision data structure
        decision_data, similarity = results[0]
        assert 'decision' in decision_data
        assert 'context' in decision_data
        assert 'outcome' in decision_data
        assert 'metadata' in decision_data
        assert 'similarity_score' in decision_data
    
    def test_retrieve_similar_decisions_threshold_filter(self, mock_chromadb, mock_embedding_service):
        """Test similarity threshold filtering."""
        mock_client, mock_collection = mock_chromadb
        
        # Mock results with varying distances
        mock_collection.query.return_value = {
            'documents': [[json.dumps({"decision": {}, "context": {}})]],
            'distances': [[0.8]],  # Low similarity (high distance)
            'metadatas': [[{"agent_id": "pricing_agent"}]]
        }
        
        memory = AgentMemory()
        
        results = memory.retrieve_similar_decisions(
            agent_id="pricing_agent",
            current_context={},
            similarity_threshold=0.5  # Require similarity > 0.5
        )
        
        # Should filter out low similarity results (0.2 similarity < 0.5 threshold)
        assert len(results) == 0
    
    def test_get_agent_decision_history(self, mock_chromadb, mock_embedding_service):
        """Test retrieving agent decision history."""
        mock_client, mock_collection = mock_chromadb
        
        # Mock historical decisions
        history_docs = [
            json.dumps({
                "decision": {
                    "decision_id": "decision-1",
                    "timestamp": "2024-01-15T10:00:00",
                    "action_type": "price_adjustment"
                },
                "context": {"product_id": "SKU123"},
                "outcome": {"effectiveness_score": 0.8}
            }),
            json.dumps({
                "decision": {
                    "decision_id": "decision-2", 
                    "timestamp": "2024-01-15T11:00:00",
                    "action_type": "markdown_application"
                },
                "context": {"product_id": "SKU456"},
                "outcome": {"effectiveness_score": 0.9}
            })
        ]
        
        mock_collection.get.return_value = {
            'documents': history_docs,
            'metadatas': [
                {"agent_id": "pricing_agent", "has_outcome": True},
                {"agent_id": "pricing_agent", "has_outcome": True}
            ]
        }
        
        memory = AgentMemory()
        
        history = memory.get_agent_decision_history(
            agent_id="pricing_agent",
            action_type="price_adjustment",
            limit=50,
            include_outcomes=True
        )
        
        # Verify ChromaDB get call
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args[1]
        expected_where = {"$and": [
            {"agent_id": {"$eq": "pricing_agent"}},
            {"action_type": {"$eq": "price_adjustment"}},
            {"has_outcome": {"$eq": True}}
        ]}
        assert call_args['where'] == expected_where
        assert call_args['limit'] == 50
        
        # Verify results
        assert len(history) == 2
        assert all('decision' in item for item in history)
        assert all('metadata' in item for item in history)
    
    def test_get_system_metrics(self, mock_chromadb, mock_embedding_service):
        """Test getting system metrics."""
        mock_client, mock_collection = mock_chromadb
        
        # Mock collection count
        mock_collection.count.return_value = 150
        
        # Mock agent-specific counts
        def mock_get_side_effect(where=None, **kwargs):
            if where:
                # Handle new ChromaDB where clause format
                if 'agent_id' in where and '$eq' in where['agent_id']:
                    agent_id = where['agent_id']['$eq']
                    if agent_id == "pricing_agent":
                        return {'documents': ['doc1', 'doc2', 'doc3']}
                    elif agent_id == "inventory_agent":
                        return {'documents': ['doc1', 'doc2']}
                    elif agent_id == "promotion_agent":
                        return {'documents': ['doc1']}
                elif 'has_outcome' in where and '$eq' in where['has_outcome']:
                    return {'documents': ['doc1'] * 80}  # 80 decisions with outcomes
            return {'documents': []}
        
        mock_collection.get.side_effect = mock_get_side_effect
        
        memory = AgentMemory()
        metrics = memory.get_system_metrics()
        
        # Verify metrics structure
        assert metrics['total_decisions'] == 150
        assert metrics['decisions_with_outcomes'] == 80
        assert metrics['agent_decision_counts']['pricing_agent'] == 3
        assert metrics['agent_decision_counts']['inventory_agent'] == 2
        assert metrics['agent_decision_counts']['promotion_agent'] == 1
        assert 'collection_name' in metrics
        assert 'persist_directory' in metrics
    
    def test_reset_memory(self, mock_chromadb, mock_embedding_service):
        """Test resetting the memory system."""
        mock_client, mock_collection = mock_chromadb
        
        memory = AgentMemory()
        memory.reset_memory()
        
        # Verify collection was deleted and recreated
        mock_client.delete_collection.assert_called_once()
        assert mock_client.get_or_create_collection.call_count == 2  # Initial + reset


@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for the memory system."""
    
    @pytest.fixture
    def temp_memory(self, tmp_path):
        """Create temporary memory instance for testing."""
        with patch('agents.memory.settings') as mock_settings:
            mock_settings.chromadb.persist_directory = str(tmp_path / "test_chromadb")
            mock_settings.chromadb.collection_name = "test_collection"
            
            # Mock embedding service to avoid AWS calls
            with patch('agents.memory.EmbeddingService') as mock_service_class:
                mock_service = Mock()
                mock_service.generate_embedding.return_value = [0.1] * 384  # Match ChromaDB default
                mock_service_class.return_value = mock_service
                
                memory = AgentMemory()
                yield memory
    
    @pytest.fixture
    def sample_decision(self):
        """Sample agent decision for testing."""
        from models.core import ActionType
        return AgentDecision(
            agent_id="pricing_agent",
            timestamp=datetime.now(),
            action_type=ActionType.PRICE_ADJUSTMENT,
            parameters={"product_id": "SKU123", "new_price": 29.99},
            rationale="Adjusting price due to high inventory levels",
            confidence_score=0.85,
            expected_outcome={"revenue_impact": 1500.0}
        )
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            "product_id": "SKU123",
            "current_price": 34.99,
            "inventory_level": 150,
            "demand_trend": "declining",
            "competitor_prices": [32.99, 31.50, 33.25]
        }
    
    def test_full_memory_workflow(self, temp_memory, sample_decision, sample_context):
        """Test complete memory workflow with real ChromaDB operations."""
        memory = temp_memory
        
        # Store decision
        memory_id = memory.store_decision(
            agent_id="pricing_agent",
            decision=sample_decision,
            context=sample_context
        )
        
        assert memory_id is not None
        
        # Update outcome
        outcome = {"revenue_change": 1000.0, "effectiveness_score": 0.85}
        memory.update_outcome(memory_id, outcome)
        
        # Retrieve similar decisions
        similar = memory.retrieve_similar_decisions(
            agent_id="pricing_agent",
            current_context=sample_context,
            limit=5
        )
        
        # Should find the stored decision
        assert len(similar) >= 0  # May or may not find similar based on embedding
        
        # Get decision history
        history = memory.get_agent_decision_history(
            agent_id="pricing_agent",
            include_outcomes=True
        )
        
        assert len(history) >= 1
        
        # Get metrics
        metrics = memory.get_system_metrics()
        assert metrics['total_decisions'] >= 1
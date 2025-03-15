import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from datetime import datetime, timedelta

from ptolemy.context_engine.retrieval import ContextRetriever


@pytest.fixture
def mock_context_engine():
    """Create a mock context engine for testing."""
    mock_engine = MagicMock()
    
    # Mock embedding manager
    mock_engine.embedding_manager = MagicMock()
    mock_engine.embedding_manager.initialized = True
    mock_engine.embedding_manager.embed_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    mock_engine.embedding_manager.calculate_similarity = AsyncMock(return_value=0.75)
    
    # Mock database methods
    mock_engine.get_all_relationships = AsyncMock(return_value=[
        {"id": "rel1", "source_entity": "ClassA", "target_entity": "ClassB", "relationship_type": "inherits"},
        {"id": "rel2", "source_entity": "ModuleX", "target_entity": "ModuleY", "relationship_type": "imports"}
    ])
    
    mock_engine.get_all_patterns = AsyncMock(return_value=[
        {"id": "pat1", "pattern_name": "Factory Pattern", "description": "A factory pattern implementation"},
        {"id": "pat2", "pattern_name": "Observer Pattern", "description": "An observer pattern implementation"}
    ])
    
    mock_engine.get_all_insights = AsyncMock(return_value=[
        {"id": "ins1", "insight_name": "Performance Bottleneck", "content": "Database query causing bottleneck"},
        {"id": "ins2", "insight_name": "Security Issue", "content": "Potential SQL injection vulnerability"}
    ])
    
    # Mock relevance scorer
    mock_engine.relevance_scorer = MagicMock()
    mock_engine.relevance_scorer.score_context_items = AsyncMock(
        side_effect=lambda items, query=None: [
            {**item, "relevance_score": 0.8 if "Factory" in str(item) or "Performance" in str(item) else 0.4} 
            for item in items
        ]
    )
    
    return mock_engine


@pytest.fixture
def context_retriever(mock_context_engine):
    """Create a ContextRetriever instance for testing."""
    return ContextRetriever(mock_context_engine)


@pytest.mark.asyncio
async def test_get_relationships_by_similarity(context_retriever, mock_context_engine):
    """Test retrieving relationships by similarity to a query."""
    relationships = await context_retriever.get_relationships_by_similarity("class inheritance", limit=5)
    
    # Check that we got relationships
    assert len(relationships) > 0
    
    # Check that mock methods were called correctly
    mock_context_engine.get_all_relationships.assert_called_once()
    mock_context_engine.relevance_scorer.score_context_items.assert_called()
    
    # Test with entity filter
    filtered_relationships = await context_retriever.get_relationships_by_similarity(
        "module dependencies", 
        entity_filter="ModuleX",
        limit=5
    )
    assert len(filtered_relationships) <= len(relationships)


@pytest.mark.asyncio
async def test_get_patterns_by_similarity(context_retriever, mock_context_engine):
    """Test retrieving patterns by similarity to a query."""
    patterns = await context_retriever.get_patterns_by_similarity("factory creation pattern", limit=5)
    
    # Check that we got patterns
    assert len(patterns) > 0
    
    # Check that mock methods were called correctly
    mock_context_engine.get_all_patterns.assert_called_once()
    mock_context_engine.relevance_scorer.score_context_items.assert_called()
    
    # The factory pattern should be first due to our mock scoring
    assert "Factory" in patterns[0]["pattern_name"]


@pytest.mark.asyncio
async def test_get_insights_by_similarity(context_retriever, mock_context_engine):
    """Test retrieving insights by similarity to a query."""
    insights = await context_retriever.get_insights_by_similarity("performance issues", limit=5)
    
    # Check that we got insights
    assert len(insights) > 0
    
    # Check that mock methods were called correctly
    mock_context_engine.get_all_insights.assert_called_once()
    mock_context_engine.relevance_scorer.score_context_items.assert_called()
    
    # The performance bottleneck should be first due to our mock scoring
    assert "Performance" in insights[0]["insight_name"]


@pytest.mark.asyncio
async def test_get_context_by_similarity(context_retriever, mock_context_engine):
    """Test retrieving combined context by similarity to a query."""
    context = await context_retriever.get_context_by_similarity("design patterns and performance", limit=5)
    
    # Check that we got context items
    assert len(context) > 0
    
    # Check that mock methods were called correctly
    mock_context_engine.get_all_relationships.assert_called()
    mock_context_engine.get_all_patterns.assert_called()
    mock_context_engine.get_all_insights.assert_called()
    
    # Test with type filter
    filtered_context = await context_retriever.get_context_by_similarity(
        "design patterns and performance",
        type_filter=["pattern"],
        limit=5
    )
    assert len(filtered_context) <= len(context)
    for item in filtered_context:
        assert item["item_type"] == "pattern"


@pytest.mark.asyncio
async def test_get_related_entities(context_retriever, mock_context_engine):
    """Test retrieving entities related to a given entity."""
    related_entities = await context_retriever.get_related_entities("ClassA")
    
    # Check that we got related entities
    assert len(related_entities) > 0
    assert "ClassB" in [entity["entity_name"] for entity in related_entities]
    
    # Check relationship types
    for entity in related_entities:
        assert "relationship_type" in entity
        assert "direction" in entity


@pytest.mark.asyncio
async def test_get_entity_context(context_retriever, mock_context_engine):
    """Test retrieving context for a specific entity."""
    entity_context = await context_retriever.get_entity_context("ClassA")
    
    # Check that we got context
    assert len(entity_context) > 0
    
    # Check that we have different types of context
    context_types = [item["item_type"] for item in entity_context if "item_type" in item]
    assert len(set(context_types)) > 0


@pytest.mark.asyncio
async def test_search_context(context_retriever, mock_context_engine):
    """Test searching context with keyword matching."""
    search_results = await context_retriever.search_context("factory pattern")
    
    # Check that we got search results
    assert len(search_results) > 0
    
    # Check that we found the factory pattern
    pattern_results = [item for item in search_results if item.get("item_type") == "pattern"]
    assert any("Factory" in item.get("pattern_name", "") for item in pattern_results)
    
    # Test with type filter
    filtered_results = await context_retriever.search_context(
        "pattern",
        type_filter=["pattern"]
    )
    assert all(item.get("item_type") == "pattern" for item in filtered_results)


@pytest.mark.asyncio
async def test_get_context_timeline(context_retriever, mock_context_engine):
    """Test retrieving context timeline."""
    # Mock the get methods to return items with timestamps
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    last_week = now - timedelta(days=7)
    
    mock_context_engine.get_all_relationships.return_value = [
        {"id": "rel1", "source_entity": "ClassA", "target_entity": "ClassB", "timestamp": now.isoformat()},
        {"id": "rel2", "source_entity": "ModuleX", "target_entity": "ModuleY", "timestamp": last_week.isoformat()}
    ]
    
    mock_context_engine.get_all_patterns.return_value = [
        {"id": "pat1", "pattern_name": "Factory Pattern", "timestamp": yesterday.isoformat()},
    ]
    
    timeline = await context_retriever.get_context_timeline(days=10)
    
    # Check that we got timeline items
    assert len(timeline) > 0
    
    # Check that items are sorted by timestamp
    for i in range(len(timeline) - 1):
        assert timeline[i]["timestamp"] >= timeline[i + 1]["timestamp"]
    
    # Test with entity filter
    filtered_timeline = await context_retriever.get_context_timeline(
        days=10,
        entity_filter="ClassA"
    )
    assert len(filtered_timeline) <= len(timeline)
    for item in filtered_timeline:
        assert "ClassA" in str(item)


@pytest.mark.asyncio
async def test_get_most_relevant_context(context_retriever, mock_context_engine):
    """Test retrieving most relevant context."""
    relevant_context = await context_retriever.get_most_relevant_context(limit=5)
    
    # Check that we got context items
    assert len(relevant_context) > 0
    
    # Check that items are sorted by relevance
    for i in range(len(relevant_context) - 1):
        assert relevant_context[i].get("relevance_score", 0) >= relevant_context[i + 1].get("relevance_score", 0)
    
    # Test with type filter
    filtered_context = await context_retriever.get_most_relevant_context(
        type_filter=["insight"],
        limit=5
    )
    assert all(item.get("item_type") == "insight" for item in filtered_context)

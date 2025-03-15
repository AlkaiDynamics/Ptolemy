import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from ptolemy.context_engine.analyzers.relevance import RelevanceScorer


@pytest.fixture
def mock_context_engine():
    """Create a mock context engine for testing."""
    mock_engine = MagicMock()
    mock_engine.embedding_manager = MagicMock()
    mock_engine.embedding_manager.initialized = True
    mock_engine.embedding_manager.calculate_similarity = AsyncMock(return_value=0.75)
    
    # Mock database update methods
    mock_engine.update_relationship = AsyncMock()
    mock_engine.update_pattern = AsyncMock()
    mock_engine.update_insight = AsyncMock()
    
    return mock_engine


@pytest.fixture
def relevance_scorer(mock_context_engine):
    """Create a RelevanceScorer instance for testing."""
    return RelevanceScorer(mock_context_engine)


@pytest.fixture
def sample_items():
    """Create sample context items for testing."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    last_week = now - timedelta(days=7)
    last_month = now - timedelta(days=30)
    
    return [
        {
            "id": "rel1",
            "item_type": "relationship",
            "source_entity": "ClassA",
            "target_entity": "ClassB",
            "relationship_type": "inherits",
            "description": "ClassA inherits from ClassB",
            "timestamp": now.isoformat(),
            "access_count": 10,
            "relevance": 0.6
        },
        {
            "id": "pat1",
            "item_type": "pattern",
            "pattern_name": "Factory Pattern",
            "implementation": "class Factory:\n    def create(self):\n        pass",
            "description": "A factory pattern implementation",
            "timestamp": yesterday.isoformat(),
            "access_count": 5,
            "relevance": 0.5
        },
        {
            "id": "ins1",
            "item_type": "insight",
            "insight_name": "Performance Bottleneck",
            "content": "The database query is causing a performance bottleneck",
            "timestamp": last_week.isoformat(),
            "access_count": 2,
            "relevance": 0.7
        },
        {
            "id": "rel2",
            "item_type": "relationship",
            "source_entity": "ModuleX",
            "target_entity": "ModuleY",
            "relationship_type": "imports",
            "description": "ModuleX imports ModuleY",
            "timestamp": last_month.isoformat(),
            "access_count": 1,
            "relevance": 0.4
        }
    ]


@pytest.mark.asyncio
async def test_score_context_item(relevance_scorer, sample_items):
    """Test scoring a single context item."""
    # Test scoring without a query
    score = await relevance_scorer.score_context_item(sample_items[0])
    assert 0.0 <= score <= 1.0
    
    # Test scoring with a query
    score_with_query = await relevance_scorer.score_context_item(sample_items[0], "ClassA inheritance")
    assert 0.0 <= score_with_query <= 1.0
    
    # Score should be cached
    cache_key = f"{sample_items[0]['id']}-ClassA inheritance"
    assert cache_key in relevance_scorer.cache
    
    # Test with pre-computed similarity score
    item_with_similarity = sample_items[0].copy()
    item_with_similarity["similarity_score"] = 0.9
    score = await relevance_scorer.score_context_item(item_with_similarity, "test query")
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_score_context_items(relevance_scorer, sample_items):
    """Test scoring multiple context items."""
    scored_items = await relevance_scorer.score_context_items(sample_items)
    
    # Check that all items have scores
    assert len(scored_items) == len(sample_items)
    for item in scored_items:
        assert "relevance_score" in item
        assert 0.0 <= item["relevance_score"] <= 1.0
    
    # Check that items are sorted by score
    for i in range(len(scored_items) - 1):
        assert scored_items[i]["relevance_score"] >= scored_items[i + 1]["relevance_score"]
    
    # Test with query
    scored_items_with_query = await relevance_scorer.score_context_items(sample_items, "factory pattern")
    assert len(scored_items_with_query) == len(sample_items)


@pytest.mark.asyncio
async def test_filter_by_relevance(relevance_scorer, sample_items):
    """Test filtering items by relevance threshold."""
    # Set a high threshold to filter out some items
    filtered_items = await relevance_scorer.filter_by_relevance(sample_items, threshold=0.7)
    assert len(filtered_items) <= len(sample_items)
    
    # Check that all items meet the threshold
    for item in filtered_items:
        assert item["relevance_score"] >= 0.7
    
    # Test with query
    filtered_items_with_query = await relevance_scorer.filter_by_relevance(
        sample_items, query="performance issues", threshold=0.5
    )
    assert len(filtered_items_with_query) <= len(sample_items)


def test_calculate_recency_score(relevance_scorer, sample_items):
    """Test recency score calculation."""
    # Recent item should have high score
    recent_score = relevance_scorer._calculate_recency_score(sample_items[0])
    assert 0.9 <= recent_score <= 1.0
    
    # Older item should have lower score
    old_score = relevance_scorer._calculate_recency_score(sample_items[3])
    assert old_score < 0.5
    
    # Test with invalid timestamp
    item_with_invalid_timestamp = sample_items[0].copy()
    item_with_invalid_timestamp["timestamp"] = "invalid-date"
    invalid_score = relevance_scorer._calculate_recency_score(item_with_invalid_timestamp)
    assert invalid_score == 0.5
    
    # Test with missing timestamp
    item_without_timestamp = sample_items[0].copy()
    del item_without_timestamp["timestamp"]
    missing_score = relevance_scorer._calculate_recency_score(item_without_timestamp)
    assert missing_score == 0.5


def test_calculate_frequency_score(relevance_scorer, sample_items):
    """Test frequency score calculation."""
    # Item with high access count should have higher score
    high_frequency_score = relevance_scorer._calculate_frequency_score(sample_items[0])
    low_frequency_score = relevance_scorer._calculate_frequency_score(sample_items[3])
    assert high_frequency_score > low_frequency_score
    
    # Test with zero access count
    item_with_zero_access = sample_items[0].copy()
    item_with_zero_access["access_count"] = 0
    zero_score = relevance_scorer._calculate_frequency_score(item_with_zero_access)
    assert zero_score == 0.0
    
    # Test with missing access count
    item_without_access = sample_items[0].copy()
    del item_without_access["access_count"]
    missing_score = relevance_scorer._calculate_frequency_score(item_without_access)
    assert missing_score == 0.0


@pytest.mark.asyncio
async def test_calculate_similarity_score(relevance_scorer, sample_items, mock_context_engine):
    """Test similarity score calculation."""
    # Test with valid embedding manager
    similarity_score = await relevance_scorer._calculate_similarity_score(
        sample_items[0], "ClassA inherits from ClassB"
    )
    assert similarity_score == 0.75  # From our mock
    
    # Test with no embedding manager
    relevance_scorer.context_engine.embedding_manager.initialized = False
    no_embedding_score = await relevance_scorer._calculate_similarity_score(
        sample_items[0], "ClassA inherits from ClassB"
    )
    assert no_embedding_score == 0.5  # Default score
    
    # Restore embedding manager for other tests
    relevance_scorer.context_engine.embedding_manager.initialized = True


def test_get_item_text(relevance_scorer, sample_items):
    """Test extracting text from context items."""
    # Test relationship item
    relationship_text = relevance_scorer._get_item_text(sample_items[0])
    assert "ClassA" in relationship_text
    assert "ClassB" in relationship_text
    assert "inherits" in relationship_text
    
    # Test pattern item
    pattern_text = relevance_scorer._get_item_text(sample_items[1])
    # Check that the text contains parts of the pattern item
    assert "factory pattern implementation" in pattern_text.lower()
    assert "class Factory" in pattern_text
    
    # Test insight item
    insight_text = relevance_scorer._get_item_text(sample_items[2])
    assert "database query" in insight_text.lower()
    assert "performance bottleneck" in insight_text.lower()
    
    # Test with metadata
    item_with_metadata = sample_items[0].copy()
    item_with_metadata["metadata"] = {
        "file_path": "/path/to/file.py",
        "line_number": 42  # Non-string value should be ignored
    }
    metadata_text = relevance_scorer._get_item_text(item_with_metadata)
    assert "/path/to/file.py" in metadata_text


@pytest.mark.asyncio
async def test_update_relevance_scores(relevance_scorer, sample_items, mock_context_engine):
    """Test updating relevance scores in the database."""
    await relevance_scorer.update_relevance_scores(sample_items)
    
    # Check that update methods were called for each item type
    mock_context_engine.update_relationship.assert_called()
    mock_context_engine.update_pattern.assert_called()
    mock_context_engine.update_insight.assert_called()
    
    # Test with query
    await relevance_scorer.update_relevance_scores(sample_items, "test query")
    
    # Test with item missing ID
    item_without_id = sample_items[0].copy()
    del item_without_id["id"]
    await relevance_scorer.update_relevance_scores([item_without_id])  # Should not raise an error


def test_clear_cache(relevance_scorer, sample_items):
    """Test clearing the score cache."""
    # Add some items to the cache
    relevance_scorer.cache = {
        "item1-query1": 0.8,
        "item2-query2": 0.6
    }
    
    # Clear the cache
    relevance_scorer.clear_cache()
    
    # Cache should be empty
    assert len(relevance_scorer.cache) == 0

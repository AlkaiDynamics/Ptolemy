
from typing import Dict, List, Optional, Any, Union, Tuple
import re
import math
from datetime import datetime
from loguru import logger

class RelevanceScorer:
    """
    Scores the relevance of context items based on various factors.
    Used to prioritize the most relevant context for AI models.
    """
    
    def __init__(self, context_engine):
        self.context_engine = context_engine
        self.recency_weight = 0.3  # Weight for recency factor
        self.frequency_weight = 0.2  # Weight for frequency factor
        self.similarity_weight = 0.5  # Weight for semantic similarity factor
        self.cache = {}  # Cache for computed scores
    
    async def score_context_item(self, item: Dict[str, Any], query: Optional[str] = None) -> float:
        """
        Score a context item based on various relevance factors.
        
        Args:
            item: The context item to score
            query: Optional query for semantic similarity scoring
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Check if we have a cached score for this item and query
            cache_key = f"{item.get('id', '')}-{query or ''}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Start with base score
            base_score = item.get("relevance", 0.5)
            
            # Calculate recency score
            recency_score = self._calculate_recency_score(item)
            
            # Calculate frequency score
            frequency_score = self._calculate_frequency_score(item)
            
            # Calculate similarity score if query is provided
            similarity_score = 0.0
            if query and self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                similarity_score = await self._calculate_similarity_score(item, query)
            elif "similarity_score" in item:
                # Use pre-computed similarity score if available
                similarity_score = item["similarity_score"]
            
            # Combine scores with weights
            if query:
                # If we have a query, prioritize similarity
                final_score = (
                    base_score * 0.2 +
                    recency_score * self.recency_weight * 0.3 +
                    frequency_score * self.frequency_weight * 0.2 +
                    similarity_score * self.similarity_weight * 0.5
                )
            else:
                # Without a query, rely more on base score and recency
                final_score = (
                    base_score * 0.4 +
                    recency_score * self.recency_weight * 0.4 +
                    frequency_score * self.frequency_weight * 0.2
                )
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            # Cache the score
            self.cache[cache_key] = final_score
            
            return final_score
        except Exception as e:
            logger.error(f"Error scoring context item: {str(e)}")
            return item.get("relevance", 0.5)  # Fall back to base score
    
    async def score_context_items(self, items: List[Dict[str, Any]], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Score and sort a list of context items based on relevance.
        
        Args:
            items: List of context items to score
            query: Optional query for semantic similarity scoring
            
        Returns:
            List of items with scores, sorted by relevance
        """
        try:
            # Score each item
            scored_items = []
            for item in items:
                score = await self.score_context_item(item, query)
                
                # Create a copy with the score
                scored_item = item.copy()
                scored_item["relevance_score"] = score
                scored_items.append(scored_item)
            
            # Sort by relevance score
            scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return scored_items
        except Exception as e:
            logger.error(f"Error scoring context items: {str(e)}")
            return items  # Return original items if error
    
    async def filter_by_relevance(self, items: List[Dict[str, Any]], query: Optional[str] = None, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter context items based on relevance threshold.
        
        Args:
            items: List of context items to filter
            query: Optional query for semantic similarity scoring
            threshold: Minimum relevance score threshold
            
        Returns:
            List of items that meet the relevance threshold
        """
        try:
            # Score items
            scored_items = await self.score_context_items(items, query)
            
            # Filter by threshold
            filtered_items = [item for item in scored_items if item.get("relevance_score", 0) >= threshold]
            
            return filtered_items
        except Exception as e:
            logger.error(f"Error filtering context items: {str(e)}")
            return items  # Return original items if error
    
    def _calculate_recency_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate recency score based on item timestamp.
        More recent items get higher scores.
        
        Args:
            item: The context item
            
        Returns:
            Recency score between 0.0 and 1.0
        """
        try:
            # Get timestamp from item
            timestamp = item.get("timestamp")
            if not timestamp:
                return 0.5  # Default score if no timestamp
            
            # Convert timestamp to datetime if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    return 0.5  # Default score if invalid timestamp
            
            # Calculate time difference in days
            now = datetime.now()
            if timestamp.tzinfo:
                now = datetime.now(timestamp.tzinfo)
            
            time_diff = (now - timestamp).total_seconds() / (24 * 3600)  # Convert to days
            
            # Calculate recency score (exponential decay)
            # Items from today get a score close to 1.0
            # Items from a week ago get a score around 0.5
            # Items from a month ago get a score close to 0.0
            recency_score = math.exp(-0.1 * time_diff)
            
            return max(0.0, min(1.0, recency_score))
        except Exception as e:
            logger.error(f"Error calculating recency score: {str(e)}")
            return 0.5  # Default score if error
    
    def _calculate_frequency_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate frequency score based on item access count.
        More frequently accessed items get higher scores.
        
        Args:
            item: The context item
            
        Returns:
            Frequency score between 0.0 and 1.0
        """
        try:
            # Get access count from item
            access_count = item.get("access_count", 0)
            
            # Calculate frequency score (logarithmic scale)
            # 0 accesses = 0.0
            # 1 access = 0.1
            # 10 accesses = 0.5
            # 100 accesses = 0.9
            if access_count <= 0:
                return 0.0
            
            frequency_score = min(1.0, 0.3 * math.log10(access_count + 1))
            
            return frequency_score
        except Exception as e:
            logger.error(f"Error calculating frequency score: {str(e)}")
            return 0.0  # Default score if error
    
    async def _calculate_similarity_score(self, item: Dict[str, Any], query: str) -> float:
        """
        Calculate semantic similarity score between item and query.
        
        Args:
            item: The context item
            query: The query text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Check if embedding manager is available
            if not self.context_engine.embedding_manager or not self.context_engine.embedding_manager.initialized:
                return 0.5  # Default score if no embedding manager
            
            # Get item text for embedding
            item_text = self._get_item_text(item)
            if not item_text:
                return 0.0
            
            # Calculate similarity using embedding manager
            similarity = await self.context_engine.embedding_manager.calculate_similarity(query, item_text)
            
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity score: {str(e)}")
            return 0.5  # Default score if error
    
    def _get_item_text(self, item: Dict[str, Any]) -> str:
        """
        Extract text from a context item for embedding.
        
        Args:
            item: The context item
            
        Returns:
            Text representation of the item
        """
        texts = []
        
        # Add name/title if available
        if "name" in item:
            texts.append(item["name"])
        elif "title" in item:
            texts.append(item["title"])
        
        # Add description if available
        if "description" in item:
            texts.append(item["description"])
        
        # Add content if available
        if "content" in item:
            texts.append(item["content"])
        
        # Add implementation if available (for patterns)
        if "implementation" in item:
            texts.append(item["implementation"])
        
        # Add source and target entities for relationships
        if "source_entity" in item and "target_entity" in item:
            texts.append(f"{item['source_entity']} {item.get('relationship_type', '')} {item['target_entity']}")
        
        # Add metadata values if available
        if "metadata" in item and isinstance(item["metadata"], dict):
            for key, value in item["metadata"].items():
                if isinstance(value, str):
                    texts.append(value)
        
        return " ".join(texts)
    
    async def update_relevance_scores(self, items: List[Dict[str, Any]], query: Optional[str] = None) -> None:
        """
        Update the relevance scores of context items in the database.
        
        Args:
            items: List of context items to update
            query: Optional query for semantic similarity scoring
        """
        try:
            for item in items:
                # Calculate new relevance score
                score = await self.score_context_item(item, query)
                
                # Update item in database based on type
                item_id = item.get("id")
                if not item_id:
                    continue
                
                item_type = item.get("item_type", "")
                
                if item_type == "relationship" or "source_entity" in item:
                    # Update relationship
                    await self.context_engine.update_relationship(
                        item_id, {"relevance": score}
                    )
                elif item_type == "pattern" or "pattern_name" in item:
                    # Update pattern
                    await self.context_engine.update_pattern(
                        item_id, {"relevance": score}
                    )
                elif item_type == "insight" or "insight_name" in item:
                    # Update insight
                    await self.context_engine.update_insight(
                        item_id, {"relevance": score}
                    )
            
            logger.info(f"Updated relevance scores for {len(items)} items")
        except Exception as e:
            logger.error(f"Error updating relevance scores: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the score cache."""
        self.cache = {}
        logger.info("Cleared relevance score cache")
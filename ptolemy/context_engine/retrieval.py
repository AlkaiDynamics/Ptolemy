# ptolemy/context_engine/retrieval.py
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger
import json
from datetime import datetime, timedelta

class ContextRetriever:
    """
    Enhances retrieval capabilities with semantic search features,
    allowing for more intelligent context retrieval based on embeddings.
    """
    
    def __init__(self, context_engine):
        self.context_engine = context_engine
        self.default_limit = 10
    
    async def get_relationships(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relationships with enhanced filtering and semantic search.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of relationships matching the filters
        """
        filters = filters or {}
        
        # Check if we need to do a semantic search
        query = filters.pop("query", None)
        limit = filters.pop("limit", self.default_limit)
        
        try:
            # If we have a query and embeddings are available, do semantic search
            if query and self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                return await self._get_relationships_by_similarity(query, filters, limit)
            else:
                # Otherwise, fall back to database/file retrieval
                return await self._get_relationships_by_filters(filters, limit)
        except Exception as e:
            logger.error(f"Error retrieving relationships: {str(e)}")
            # Fall back to original implementation
            return await self.context_engine.get_relationships(filters)
    
    async def get_patterns(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve patterns with enhanced filtering and semantic search.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of patterns matching the filters
        """
        filters = filters or {}
        
        # Check if we need to do a semantic search
        query = filters.pop("query", None)
        limit = filters.pop("limit", self.default_limit)
        
        try:
            # If we have a query and embeddings are available, do semantic search
            if query and self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                return await self._get_patterns_by_similarity(query, filters, limit)
            else:
                # Otherwise, fall back to database/file retrieval
                return await self._get_patterns_by_filters(filters, limit)
        except Exception as e:
            logger.error(f"Error retrieving patterns: {str(e)}")
            # Fall back to original implementation
            return await self.context_engine.get_patterns(filters)
    
    async def get_insights(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve insights with enhanced filtering and semantic search.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of insights matching the filters
        """
        filters = filters or {}
        
        # Check if we need to do a semantic search
        query = filters.pop("query", None)
        limit = filters.pop("limit", self.default_limit)
        
        try:
            # If we have a query and embeddings are available, do semantic search
            if query and self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                return await self._get_insights_by_similarity(query, filters, limit)
            else:
                # Otherwise, fall back to database/file retrieval
                return await self._get_insights_by_filters(filters, limit)
        except Exception as e:
            logger.error(f"Error retrieving insights: {str(e)}")
            # Fall back to original implementation
            return await self.context_engine.get_insights(filters)
    
    async def get_context_by_similarity(self, query: str, limit: int = 5, type_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve context items (relationships, patterns, insights) by semantic similarity.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            type_filter: Optional list of item types to include (e.g. ['relationship', 'pattern'])
            
        Returns:
            A list of context items with relevance scores
        """
        try:
            # Get all context items
            context_items = []
            
            # Get items of each type if not filtered out
            if not type_filter or "relationship" in type_filter:
                relationships = await self.context_engine.get_all_relationships()
                for rel in relationships:
                    rel["item_type"] = "relationship"
                    context_items.append(rel)
            
            if not type_filter or "pattern" in type_filter:
                patterns = await self.context_engine.get_all_patterns()
                for pat in patterns:
                    pat["item_type"] = "pattern"
                    context_items.append(pat)
            
            if not type_filter or "insight" in type_filter:
                insights = await self.context_engine.get_all_insights()
                for ins in insights:
                    ins["item_type"] = "insight"
                    context_items.append(ins)
            
            # Use relevance scorer to score and filter items
            if hasattr(self.context_engine, "relevance_scorer") and self.context_engine.relevance_scorer:
                scored_items = await self.context_engine.relevance_scorer.score_context_items(context_items, query)
                scored_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return scored_items[:limit]
            else:
                # Score items using embeddings if available
                if self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                    query_embedding = await self.context_engine.embedding_manager.embed_text(query)
                    
                    # Score each item and add a similarity score
                    for item in context_items:
                        item_text = self._get_item_text(item)
                        item_embedding = await self.context_engine.embedding_manager.embed_text(item_text)
                        similarity = await self.context_engine.embedding_manager.calculate_similarity(
                            query_embedding, item_embedding
                        )
                        item["similarity_score"] = similarity
                    
                    # Sort by similarity and limit results
                    context_items.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                    return context_items[:limit]
                else:
                    # If no embedding manager, do simple keyword matching
                    return await self.search_context(query, limit=limit, type_filter=type_filter)
        except Exception as e:
            logger.error(f"Error in semantic context retrieval: {str(e)}")
            return []
    
    async def get_relationships_by_similarity(
        self, query: str, limit: int = 5, entity_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relationships by semantic similarity to the query.
        
        Args:
            query: The search query
            limit: Maximum number of relationships to return
            entity_filter: Optional entity name to filter by
            
        Returns:
            A list of relationships with relevance scores
        """
        try:
            # Get all relationships
            relationships = await self.context_engine.get_all_relationships()
            
            # Filter by entity if specified
            if entity_filter:
                relationships = [
                    rel for rel in relationships 
                    if (rel.get("source_entity") == entity_filter or 
                        rel.get("target_entity") == entity_filter)
                ]
            
            # Add item type for scoring
            for rel in relationships:
                rel["item_type"] = "relationship"
            
            # Use relevance scorer to score items
            if hasattr(self.context_engine, "relevance_scorer") and self.context_engine.relevance_scorer:
                scored_items = await self.context_engine.relevance_scorer.score_context_items(relationships, query)
                scored_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return scored_items[:limit]
            else:
                # Score using embeddings if available
                if self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                    return await self._score_items_with_embeddings(relationships, query, limit)
                else:
                    # Sort by timestamp if available for simple recency scoring
                    relationships.sort(
                        key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                        if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                        reverse=True
                    )
                    return relationships[:limit]
        except Exception as e:
            logger.error(f"Error in semantic relationship retrieval: {str(e)}")
            return []
    
    async def get_patterns_by_similarity(
        self, query: str, limit: int = 5, pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve patterns by semantic similarity to the query.
        
        Args:
            query: The search query
            limit: Maximum number of patterns to return
            pattern_type: Optional pattern type to filter by
            
        Returns:
            A list of patterns with relevance scores
        """
        try:
            # Get all patterns
            patterns = await self.context_engine.get_all_patterns()
            
            # Filter by pattern type if specified
            if pattern_type:
                patterns = [pat for pat in patterns if pat.get("pattern_type") == pattern_type]
            
            # Add item type for scoring
            for pat in patterns:
                pat["item_type"] = "pattern"
            
            # Use relevance scorer to score items
            if hasattr(self.context_engine, "relevance_scorer") and self.context_engine.relevance_scorer:
                scored_items = await self.context_engine.relevance_scorer.score_context_items(patterns, query)
                scored_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return scored_items[:limit]
            else:
                # Score using embeddings if available
                if self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                    return await self._score_items_with_embeddings(patterns, query, limit)
                else:
                    # Sort by timestamp if available for simple recency scoring
                    patterns.sort(
                        key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                        if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                        reverse=True
                    )
                    return patterns[:limit]
        except Exception as e:
            logger.error(f"Error in semantic pattern retrieval: {str(e)}")
            return []
    
    async def get_insights_by_similarity(
        self, query: str, limit: int = 5, insight_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve insights by semantic similarity to the query.
        
        Args:
            query: The search query
            limit: Maximum number of insights to return
            insight_type: Optional insight type to filter by
            
        Returns:
            A list of insights with relevance scores
        """
        try:
            # Get all insights
            insights = await self.context_engine.get_all_insights()
            
            # Filter by insight type if specified
            if insight_type:
                insights = [ins for ins in insights if ins.get("insight_type") == insight_type]
            
            # Add item type for scoring
            for ins in insights:
                ins["item_type"] = "insight"
            
            # Use relevance scorer to score items
            if hasattr(self.context_engine, "relevance_scorer") and self.context_engine.relevance_scorer:
                scored_items = await self.context_engine.relevance_scorer.score_context_items(insights, query)
                scored_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                return scored_items[:limit]
            else:
                # Score using embeddings if available
                if self.context_engine.embedding_manager and self.context_engine.embedding_manager.initialized:
                    return await self._score_items_with_embeddings(insights, query, limit)
                else:
                    # Sort by timestamp if available for simple recency scoring
                    insights.sort(
                        key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                        if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                        reverse=True
                    )
                    return insights[:limit]
        except Exception as e:
            logger.error(f"Error in semantic insight retrieval: {str(e)}")
            return []
    
    async def get_related_entities(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entities that are related to the specified entity.
        
        Args:
            entity_name: The name of the entity to get relations for
            limit: Maximum number of related entities to return
            
        Returns:
            A list of related entities with relationship information
        """
        try:
            # Get relationships involving this entity
            relationships = await self.context_engine.get_all_relationships()
            entity_relationships = [
                rel for rel in relationships 
                if rel.get("source_entity") == entity_name or rel.get("target_entity") == entity_name
            ]
            
            # Extract related entities
            related_entities = []
            for rel in entity_relationships:
                related_name = rel.get("target_entity") if rel.get("source_entity") == entity_name else rel.get("source_entity")
                direction = "outgoing" if rel.get("source_entity") == entity_name else "incoming"
                
                related_entities.append({
                    "entity_name": related_name,
                    "relationship_type": rel.get("relationship_type"),
                    "direction": direction,
                    "relationship_id": rel.get("id"),
                    "description": rel.get("description", "")
                })
            
            # Sort by recency if available
            related_entities.sort(
                key=lambda x: next(
                    (datetime.fromisoformat(rel.get("timestamp", "2000-01-01T00:00:00")) 
                    for rel in entity_relationships if rel.get("id") == x.get("relationship_id")),
                    datetime(2000, 1, 1)
                ),
                reverse=True
            )
            
            return related_entities[:limit]
        except Exception as e:
            logger.error(f"Error getting related entities: {str(e)}")
            return []
    
    async def get_entity_context(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all context items related to a specific entity.
        
        Args:
            entity_name: The name of the entity to get context for
            limit: Maximum number of context items to return
            
        Returns:
            A list of context items related to the entity
        """
        try:
            context_items = []
            
            # Get relationships involving this entity
            relationships = await self.context_engine.get_all_relationships()
            entity_relationships = [
                rel for rel in relationships 
                if rel.get("source_entity") == entity_name or rel.get("target_entity") == entity_name
            ]
            
            for rel in entity_relationships:
                rel["item_type"] = "relationship"
                context_items.append(rel)
            
            # Get patterns associated with this entity
            patterns = await self.context_engine.get_all_patterns()
            entity_patterns = [
                pat for pat in patterns 
                if entity_name in json.dumps(pat)  # Simple way to check if entity is mentioned
            ]
            
            for pat in entity_patterns:
                pat["item_type"] = "pattern"
                context_items.append(pat)
            
            # Get insights related to this entity
            insights = await self.context_engine.get_all_insights()
            entity_insights = [
                ins for ins in insights 
                if entity_name in json.dumps(ins)  # Simple way to check if entity is mentioned
            ]
            
            for ins in entity_insights:
                ins["item_type"] = "insight"
                context_items.append(ins)
            
            # Sort by recency if available
            context_items.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                reverse=True
            )
            
            return context_items[:limit]
        except Exception as e:
            logger.error(f"Error getting entity context: {str(e)}")
            return []
    
    async def search_context(
        self, query: str, limit: int = 10, type_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search all context items for matching text.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            type_filter: Optional list of item types to include
            
        Returns:
            A list of context items matching the query
        """
        try:
            # Get all context items
            context_items = []
            
            # Get items of each type if not filtered out
            if not type_filter or "relationship" in type_filter:
                relationships = await self.context_engine.get_all_relationships()
                for rel in relationships:
                    rel["item_type"] = "relationship"
                    context_items.append(rel)
            
            if not type_filter or "pattern" in type_filter:
                patterns = await self.context_engine.get_all_patterns()
                for pat in patterns:
                    pat["item_type"] = "pattern"
                    context_items.append(pat)
            
            if not type_filter or "insight" in type_filter:
                insights = await self.context_engine.get_all_insights()
                for ins in insights:
                    ins["item_type"] = "insight"
                    context_items.append(ins)
            
            # Perform simple keyword matching
            query_terms = query.lower().split()
            matched_items = []
            
            for item in context_items:
                item_json = json.dumps(item).lower()
                match_score = sum(1 for term in query_terms if term in item_json)
                if match_score > 0:
                    item["match_score"] = match_score
                    matched_items.append(item)
            
            # Sort by match score
            matched_items.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            return matched_items[:limit]
        except Exception as e:
            logger.error(f"Error searching context: {str(e)}")
            return []
    
    async def get_context_timeline(self, days: int = 30, limit: int = 20, entity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a timeline of context events from the recent past.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of events to return
            entity_filter: Optional entity name to filter by
            
        Returns:
            A list of context items sorted by timestamp
        """
        try:
            # Get start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Get all context items
            context_items = []
            
            # Get relationships
            relationships = await self.context_engine.get_all_relationships()
            # Apply entity filter if specified
            if entity_filter:
                relationships = [
                    rel for rel in relationships 
                    if rel.get("source_entity") == entity_filter or rel.get("target_entity") == entity_filter
                ]
            
            for rel in relationships:
                rel["item_type"] = "relationship"
                context_items.append(rel)
            
            # Get patterns
            patterns = await self.context_engine.get_all_patterns()
            if entity_filter:
                patterns = [
                    pat for pat in patterns 
                    if entity_filter in json.dumps(pat)  # Simple way to check if entity is mentioned
                ]
                
            for pat in patterns:
                pat["item_type"] = "pattern"
                context_items.append(pat)
            
            # Get insights
            insights = await self.context_engine.get_all_insights()
            if entity_filter:
                insights = [
                    ins for ins in insights 
                    if entity_filter in json.dumps(ins)  # Simple way to check if entity is mentioned
                ]
                
            for ins in insights:
                ins["item_type"] = "insight"
                context_items.append(ins)
            
            # Filter items by date
            recent_items = []
            for item in context_items:
                if "timestamp" in item and isinstance(item["timestamp"], str):
                    try:
                        item_date = datetime.fromisoformat(item["timestamp"])
                        if item_date >= start_date:
                            recent_items.append(item)
                    except (ValueError, TypeError):
                        continue
            
            # Sort by timestamp
            recent_items.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00"))
                if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                reverse=True
            )
            
            return recent_items[:limit]
        except Exception as e:
            logger.error(f"Error getting context timeline: {str(e)}")
            return []
    
    async def get_most_relevant_context(self, query: str = "", limit: int = 5, type_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get the most relevant context for the given query, using the RelevanceScorer.
        
        Args:
            query: The query to find relevant context for
            limit: Maximum number of context items to return
            type_filter: Optional list of item types to include (e.g. ['relationship', 'pattern'])
            
        Returns:
            A list of the most relevant context items
        """
        try:
            # Get all context items
            context_items = []
            
            # Get items of each type if not filtered out
            if not type_filter or "relationship" in type_filter:
                relationships = await self.context_engine.get_all_relationships()
                for rel in relationships:
                    rel["item_type"] = "relationship"
                    context_items.append(rel)
            
            if not type_filter or "pattern" in type_filter:
                patterns = await self.context_engine.get_all_patterns()
                for pat in patterns:
                    pat["item_type"] = "pattern"
                    context_items.append(pat)
            
            if not type_filter or "insight" in type_filter:
                insights = await self.context_engine.get_all_insights()
                for ins in insights:
                    ins["item_type"] = "insight"
                    context_items.append(ins)
            
            # If no query is provided, sort by recency
            if not query:
                context_items.sort(
                    key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                    if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                    reverse=True
                )
                return context_items[:limit]
                
            # Use the relevance scorer to filter and rank items
            if hasattr(self.context_engine, "relevance_scorer") and self.context_engine.relevance_scorer:
                return await self.context_engine.relevance_scorer.filter_by_relevance(
                    context_items, query=query, threshold=0.5, limit=limit
                )
            else:
                # Fall back to simpler methods if no relevance scorer
                return await self.get_context_by_similarity(query, limit=limit, type_filter=type_filter)
        except Exception as e:
            logger.error(f"Error getting most relevant context: {str(e)}")
            return []
    
    async def _get_relationships_by_filters(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Get relationships by applying filters."""
        try:
            # Get all relationships
            all_relationships = await self.context_engine.get_all_relationships()
            
            # Apply filters
            filtered_relationships = []
            for rel in all_relationships:
                if self._matches_filters(rel, filters):
                    filtered_relationships.append(rel)
            
            # Sort by timestamp if available
            filtered_relationships.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                reverse=True
            )
            
            return filtered_relationships[:limit]
        except Exception as e:
            logger.error(f"Error retrieving relationships by filters: {str(e)}")
            return []
    
    async def _get_patterns_by_filters(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Get patterns by applying filters."""
        try:
            # Get all patterns
            all_patterns = await self.context_engine.get_all_patterns()
            
            # Apply filters
            filtered_patterns = []
            for pat in all_patterns:
                if self._matches_filters(pat, filters):
                    filtered_patterns.append(pat)
            
            # Sort by timestamp if available
            filtered_patterns.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                reverse=True
            )
            
            return filtered_patterns[:limit]
        except Exception as e:
            logger.error(f"Error retrieving patterns by filters: {str(e)}")
            return []
    
    async def _get_insights_by_filters(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Get insights by applying filters."""
        try:
            # Get all insights
            all_insights = await self.context_engine.get_all_insights()
            
            # Apply filters
            filtered_insights = []
            for ins in all_insights:
                if self._matches_filters(ins, filters):
                    filtered_insights.append(ins)
            
            # Sort by timestamp if available
            filtered_insights.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01T00:00:00")) 
                if isinstance(x.get("timestamp", ""), str) else datetime(2000, 1, 1),
                reverse=True
            )
            
            return filtered_insights[:limit]
        except Exception as e:
            logger.error(f"Error retrieving insights by filters: {str(e)}")
            return []
    
    async def _score_items_with_embeddings(self, items: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """Score items using text embeddings."""
        try:
            query_embedding = await self.context_engine.embedding_manager.embed_text(query)
            
            # Score each item and add a similarity score
            for item in items:
                item_text = self._get_item_text(item)
                item_embedding = await self.context_engine.embedding_manager.embed_text(item_text)
                similarity = await self.context_engine.embedding_manager.calculate_similarity(
                    query_embedding, item_embedding
                )
                item["similarity_score"] = similarity
            
            # Sort by similarity and limit results
            items.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return items[:limit]
        except Exception as e:
            logger.error(f"Error scoring items with embeddings: {str(e)}")
            return items[:limit]  # Return unscored items as fallback
    
    def _matches_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if an item matches all filters."""
        for key, value in filters.items():
            if key not in item or item[key] != value:
                return False
        return True
    
    def _get_item_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a context item for embedding."""
        text_parts = []
        
        # Process different item types
        if item.get("item_type") == "relationship":
            if "source_entity" in item:
                text_parts.append(str(item["source_entity"]))
            if "target_entity" in item:
                text_parts.append(str(item["target_entity"]))
            if "relationship_type" in item:
                text_parts.append(str(item["relationship_type"]))
            if "description" in item:
                text_parts.append(str(item["description"]))
        
        elif item.get("item_type") == "pattern":
            if "pattern_name" in item:
                text_parts.append(str(item["pattern_name"]))
            if "description" in item:
                text_parts.append(str(item["description"]))
            if "implementation" in item:
                text_parts.append(str(item["implementation"]))
        
        elif item.get("item_type") == "insight":
            if "insight_name" in item:
                text_parts.append(str(item["insight_name"]))
            if "content" in item:
                text_parts.append(str(item["content"]))
        
        # Add metadata if available
        if "metadata" in item and isinstance(item["metadata"], dict):
            for key, value in item["metadata"].items():
                if isinstance(value, str):
                    text_parts.append(value)
        
        # Join all parts
        return " ".join(text_parts)
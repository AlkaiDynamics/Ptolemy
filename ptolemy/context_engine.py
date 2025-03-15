import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import select, and_

from loguru import logger
from ptolemy.config import CONTEXT_DIR
from ptolemy.temporal_core import TemporalCore
from ptolemy.database import async_session, Relationship, Pattern, Insight


class ContextEngine:
    """
    Context Engine manages relationships between project entities, 
    stores reusable implementation patterns, and manages project-specific insights.
    """
    
    def __init__(self, temporal_core: TemporalCore, storage_path: Optional[Path] = None):
        self.temporal_core = temporal_core
        self.storage_path = storage_path or CONTEXT_DIR
        self.relationships_path = self.storage_path / "relationships"
        self.patterns_path = self.storage_path / "patterns"
        self.insights_path = self.storage_path / "insights"
        self.db_enabled = True
    
    async def initialize(self):
        """Initialize the Context Engine system."""
        try:
            self.relationships_path.mkdir(parents=True, exist_ok=True)
            self.patterns_path.mkdir(parents=True, exist_ok=True)
            self.insights_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Context Engine initialized with storage path: {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Context Engine: {str(e)}")
            raise
    
    async def store_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a relationship between two entities.
        
        Args:
            source_entity: The source entity
            target_entity: The target entity
            relationship_type: The type of relationship
            metadata: Optional metadata about the relationship
            
        Returns:
            The stored relationship object
        """
        metadata = metadata or {}
        relationship = {
            "id": str(uuid.uuid4()),
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relationship_type": relationship_type,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Store in file system for backward compatibility
            relationship_file_path = self.relationships_path / f"{relationship['id']}.json"
            with open(relationship_file_path, 'w') as f:
                json.dump(relationship, f, indent=2)
            
            # Store in database if enabled
            if self.db_enabled:
                async with async_session() as session:
                    db_relationship = Relationship(
                        id=relationship["id"],
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relationship_type=relationship_type,
                        meta_data=metadata,
                        timestamp=datetime.fromisoformat(relationship["timestamp"])
                    )
                    session.add(db_relationship)
                    await session.commit()
            
            # Record this as an event in the temporal core
            await self.temporal_core.record_event("relationship_created", {
                "relationship_id": relationship["id"],
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relationship_type": relationship_type
            })
            
            logger.info(f"Relationship stored: {source_entity} -> {target_entity} ({relationship_type})")
            return relationship
        except Exception as e:
            logger.error(f"Failed to store relationship: {str(e)}")
            raise
    
    async def store_pattern(
        self, 
        pattern_name: str, 
        pattern_type: str, 
        implementation: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a reusable implementation pattern.
        
        Args:
            pattern_name: The name of the pattern
            pattern_type: The type of pattern
            implementation: The implementation code or description
            metadata: Optional metadata about the pattern
            
        Returns:
            The stored pattern object
        """
        metadata = metadata or {}
        pattern = {
            "id": str(uuid.uuid4()),
            "name": pattern_name,
            "type": pattern_type,
            "implementation": implementation,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Store in file system for backward compatibility
            pattern_file_path = self.patterns_path / f"{pattern['id']}.json"
            with open(pattern_file_path, 'w') as f:
                json.dump(pattern, f, indent=2)
            
            # Store in database if enabled
            if self.db_enabled:
                async with async_session() as session:
                    db_pattern = Pattern(
                        id=pattern["id"],
                        name=pattern_name,
                        type=pattern_type,
                        implementation=implementation,
                        meta_data=metadata,
                        timestamp=datetime.fromisoformat(pattern["timestamp"])
                    )
                    session.add(db_pattern)
                    await session.commit()
            
            # Record this as an event in the temporal core
            await self.temporal_core.record_event("pattern_stored", {
                "pattern_id": pattern["id"],
                "pattern_name": pattern_name,
                "pattern_type": pattern_type
            })
            
            logger.info(f"Pattern stored: {pattern_name} ({pattern_type})")
            return pattern
        except Exception as e:
            logger.error(f"Failed to store pattern: {str(e)}")
            raise
    
    async def store_insight(
        self, 
        insight_type: str, 
        content: str, 
        relevance: float = 1.0, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a project-specific insight.
        
        Args:
            insight_type: The type of insight
            content: The content of the insight
            relevance: The relevance score (0.0 to 1.0)
            metadata: Optional metadata about the insight
            
        Returns:
            The stored insight object
        """
        metadata = metadata or {}
        insight = {
            "id": str(uuid.uuid4()),
            "type": insight_type,
            "content": content,
            "relevance": relevance,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Store in file system for backward compatibility
            insight_file_path = self.insights_path / f"{insight['id']}.json"
            with open(insight_file_path, 'w') as f:
                json.dump(insight, f, indent=2)
            
            # Store in database if enabled
            if self.db_enabled:
                async with async_session() as session:
                    db_insight = Insight(
                        id=insight["id"],
                        type=insight_type,
                        content=content,
                        relevance=relevance,
                        meta_data=metadata,
                        timestamp=datetime.fromisoformat(insight["timestamp"])
                    )
                    session.add(db_insight)
                    await session.commit()
            
            # Record this as an event in the temporal core
            await self.temporal_core.record_event("insight_stored", {
                "insight_id": insight["id"],
                "insight_type": insight_type,
                "relevance": relevance
            })
            
            logger.info(f"Insight stored: {insight_type} (relevance: {relevance})")
            return insight
        except Exception as e:
            logger.error(f"Failed to store insight: {str(e)}")
            raise
    
    async def get_relationships(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relationships, optionally filtered.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of relationships matching the filters
        """
        filters = filters or {}
        try:
            if self.db_enabled:
                # Use database query with SQLAlchemy
                async with async_session() as session:
                    query = select(Relationship)
                    
                    # Apply filters
                    filter_conditions = []
                    for key, value in filters.items():
                        if hasattr(Relationship, key):
                            filter_conditions.append(getattr(Relationship, key) == value)
                    
                    if filter_conditions:
                        query = query.where(and_(*filter_conditions))
                    
                    # Order by timestamp
                    query = query.order_by(Relationship.timestamp)
                    
                    result = await session.execute(query)
                    db_relationships = result.scalars().all()
                    
                    # Convert to dict format for API consistency
                    relationships = []
                    for db_rel in db_relationships:
                        relationships.append({
                            "id": db_rel.id,
                            "source_entity": db_rel.source_entity,
                            "target_entity": db_rel.target_entity,
                            "relationship_type": db_rel.relationship_type,
                            "meta_data": db_rel.meta_data,
                            "timestamp": db_rel.timestamp.isoformat()
                        })
                    
                    return relationships
            else:
                # Fall back to file-based retrieval
                relationships = []
                for relationship_file in self.relationships_path.glob("*.json"):
                    with open(relationship_file, 'r') as f:
                        relationship_data = json.load(f)
                    
                    # Apply filters if any
                    include_relationship = True
                    for key, value in filters.items():
                        if relationship_data.get(key) != value:
                            include_relationship = False
                            break
                    
                    if include_relationship:
                        relationships.append(relationship_data)
                
                # Sort by timestamp
                return sorted(relationships, key=lambda x: x["timestamp"])
        except Exception as e:
            logger.error(f"Failed to get relationships: {str(e)}")
            raise
    
    async def get_patterns(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve patterns, optionally filtered.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of patterns matching the filters
        """
        filters = filters or {}
        try:
            if self.db_enabled:
                # Use database query with SQLAlchemy
                async with async_session() as session:
                    query = select(Pattern)
                    
                    # Apply filters
                    filter_conditions = []
                    for key, value in filters.items():
                        if hasattr(Pattern, key):
                            filter_conditions.append(getattr(Pattern, key) == value)
                    
                    if filter_conditions:
                        query = query.where(and_(*filter_conditions))
                    
                    # Order by timestamp
                    query = query.order_by(Pattern.timestamp)
                    
                    result = await session.execute(query)
                    db_patterns = result.scalars().all()
                    
                    # Convert to dict format for API consistency
                    patterns = []
                    for db_pattern in db_patterns:
                        patterns.append({
                            "id": db_pattern.id,
                            "name": db_pattern.name,
                            "type": db_pattern.type,
                            "implementation": db_pattern.implementation,
                            "meta_data": db_pattern.meta_data,
                            "timestamp": db_pattern.timestamp.isoformat()
                        })
                    
                    return patterns
            else:
                # Fall back to file-based retrieval
                patterns = []
                for pattern_file in self.patterns_path.glob("*.json"):
                    with open(pattern_file, 'r') as f:
                        pattern_data = json.load(f)
                    
                    # Apply filters if any
                    include_pattern = True
                    for key, value in filters.items():
                        if pattern_data.get(key) != value:
                            include_pattern = False
                            break
                    
                    if include_pattern:
                        patterns.append(pattern_data)
                
                # Sort by timestamp
                return sorted(patterns, key=lambda x: x["timestamp"])
        except Exception as e:
            logger.error(f"Failed to get patterns: {str(e)}")
            raise
    
    async def get_insights(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve insights, optionally filtered.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of insights matching the filters
        """
        filters = filters or {}
        try:
            if self.db_enabled:
                # Use database query with SQLAlchemy
                async with async_session() as session:
                    query = select(Insight)
                    
                    # Apply filters
                    filter_conditions = []
                    for key, value in filters.items():
                        if hasattr(Insight, key):
                            filter_conditions.append(getattr(Insight, key) == value)
                    
                    if filter_conditions:
                        query = query.where(and_(*filter_conditions))
                    
                    # Order by timestamp and relevance
                    query = query.order_by(Insight.relevance.desc(), Insight.timestamp.desc())
                    
                    result = await session.execute(query)
                    db_insights = result.scalars().all()
                    
                    # Convert to dict format for API consistency
                    insights = []
                    for db_insight in db_insights:
                        insights.append({
                            "id": db_insight.id,
                            "type": db_insight.type,
                            "content": db_insight.content,
                            "relevance": db_insight.relevance,
                            "meta_data": db_insight.meta_data,
                            "timestamp": db_insight.timestamp.isoformat()
                        })
                    
                    return insights
            else:
                # Fall back to file-based retrieval
                insights = []
                for insight_file in self.insights_path.glob("*.json"):
                    with open(insight_file, 'r') as f:
                        insight_data = json.load(f)
                    
                    # Apply filters if any
                    include_insight = True
                    for key, value in filters.items():
                        if insight_data.get(key) != value:
                            include_insight = False
                            break
                    
                    if include_insight:
                        insights.append(insight_data)
                
                # Sort by relevance and timestamp
                return sorted(insights, key=lambda x: (-x.get("relevance", 0), x["timestamp"]), reverse=True)
        except Exception as e:
            logger.error(f"Failed to get insights: {str(e)}")
            raise
    
    async def get_model_context(self, task: str) -> str:
        """
        Generate context for a model based on the task.
        
        Args:
            task: The task description
            
        Returns:
            Context information for the model
        """
        # Get relevant patterns
        patterns = await self.get_patterns()
        pattern_context = "\n".join([f"PATTERN: {p['name']} - {p['implementation'][:100]}..." for p in patterns[:3]])
        
        # Get relevant insights
        insights = await self.get_insights()
        insight_context = "\n".join([f"INSIGHT: {i['content'][:150]}..." for i in insights[:3]])
        
        return f"CONTEXT:\n{pattern_context}\n\n{insight_context}"

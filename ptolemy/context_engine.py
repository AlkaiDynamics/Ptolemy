import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from loguru import logger
from ptolemy.config import CONTEXT_DIR
from ptolemy.temporal_core import TemporalCore


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
            relationship_file_path = self.relationships_path / f"{relationship['id']}.json"
            with open(relationship_file_path, 'w') as f:
                json.dump(relationship, f, indent=2)
            
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
            pattern_file_path = self.patterns_path / f"{pattern['id']}.json"
            with open(pattern_file_path, 'w') as f:
                json.dump(pattern, f, indent=2)
            
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
            insight_file_path = self.insights_path / f"{insight['id']}.json"
            with open(insight_file_path, 'w') as f:
                json.dump(insight, f, indent=2)
            
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
            
            return relationships
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
            
            return patterns
        except Exception as e:
            logger.error(f"Failed to get patterns: {str(e)}")
            raise
    
    async def get_insights(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve insights, optionally filtered.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            A list of insights matching the filters, sorted by relevance
        """
        filters = filters or {}
        try:
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
            
            # Sort by relevance (highest first)
            return sorted(insights, key=lambda x: x["relevance"], reverse=True)
        except Exception as e:
            logger.error(f"Failed to get insights: {str(e)}")
            raise
    
    async def get_model_context(self, task: str) -> str:
        """
        Build context for AI models based on the task.
        
        Args:
            task: The task description
            
        Returns:
            A formatted context string
        """
        # Get relevant insights and patterns
        insights = await self.get_insights()
        patterns = await self.get_patterns()
        
        # Build context string
        context_string = "# Project Context\n\n"
        
        # Add insights
        if insights:
            context_string += "## Insights\n\n"
            for insight in insights[:5]:  # Top 5 insights by relevance
                context_string += f"- {insight['content']} ({insight['type']}, relevance: {insight['relevance']})\n"
            context_string += "\n"
        
        # Add patterns
        if patterns:
            context_string += "## Implementation Patterns\n\n"
            for pattern in patterns[:3]:  # Top 3 patterns
                context_string += f"### {pattern['name']} ({pattern['type']})\n\n```\n{pattern['implementation']}\n```\n\n"
        
        return context_string

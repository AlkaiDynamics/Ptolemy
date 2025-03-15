import json
from typing import Dict, List, Optional, Any, Union

from loguru import logger

from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine


class FeedbackOrchestrator:
    """
    Feedback Orchestrator analyzes user-driven changes, continuously adjusts
    AI-generation quality, and provides adaptive responses to project needs
    and user preferences.
    """
    
    def __init__(self, temporal_core: TemporalCore, context_engine: ContextEngine):
        self.temporal_core = temporal_core
        self.context_engine = context_engine
        self.feedback_history = []
    
    async def record_feedback(
        self, 
        feedback_type: str, 
        content: str, 
        source: str, 
        target_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record feedback for later analysis.
        
        Args:
            feedback_type: The type of feedback (e.g., "code_quality", "user_preference")
            content: The feedback content
            source: The source of the feedback (e.g., "user", "automated_test")
            target_id: Optional ID of the target (e.g., event ID, pattern ID)
            metadata: Optional additional metadata
            
        Returns:
            The recorded feedback event
        """
        metadata = metadata or {}
        feedback_data = {
            "feedback_type": feedback_type,
            "content": content,
            "source": source,
            "target_id": target_id,
            "metadata": metadata
        }
        
        # Record as an event in the temporal core
        event = await self.temporal_core.record_event("feedback_received", feedback_data)
        self.feedback_history.append(event)
        
        # If this is a significant insight, store it in the context engine
        if feedback_type in ["user_preference", "architecture_feedback", "quality_standard"]:
            relevance = metadata.get("relevance", 0.8)
            await self.context_engine.store_insight(
                f"feedback_{feedback_type}",
                content,
                relevance,
                {"source": source, "event_id": event["id"]}
            )
        
        logger.info(f"Feedback recorded: {feedback_type} from {source}")
        return event
    
    async def analyze_feedback_trends(self, feedback_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trends in feedback to identify patterns.
        
        Args:
            feedback_type: Optional filter for specific feedback type
            
        Returns:
            Analysis results
        """
        filters = {}
        if feedback_type:
            filters["type"] = "feedback_received"
        
        events = await self.temporal_core.get_events(filters)
        feedback_events = [e for e in events if e["type"] == "feedback_received"]
        
        if feedback_type:
            feedback_events = [e for e in feedback_events if e["data"]["feedback_type"] == feedback_type]
        
        # Simple analysis - count by type and source
        type_counts = {}
        source_counts = {}
        
        for event in feedback_events:
            fb_type = event["data"]["feedback_type"]
            fb_source = event["data"]["source"]
            
            type_counts[fb_type] = type_counts.get(fb_type, 0) + 1
            source_counts[fb_source] = source_counts.get(fb_source, 0) + 1
        
        return {
            "total_feedback": len(feedback_events),
            "by_type": type_counts,
            "by_source": source_counts,
            "recent_feedback": feedback_events[-5:] if feedback_events else []
        }
    
    async def adjust_generation_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Adjust generation parameters based on feedback history.
        
        Args:
            model_type: The type of model to adjust parameters for
            
        Returns:
            Adjusted parameters
        """
        # Get relevant feedback for this model type
        model_feedback = []
        for event in self.feedback_history:
            if event["data"].get("metadata", {}).get("model_type") == model_type:
                model_feedback.append(event)
        
        # Default adjustments
        adjustments = {
            "temperature": 0.0,  # No change by default
            "top_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        
        # Simple heuristic adjustments based on feedback
        if model_feedback:
            # Count positive and negative feedback
            positive = sum(1 for e in model_feedback if e["data"].get("metadata", {}).get("sentiment") == "positive")
            negative = sum(1 for e in model_feedback if e["data"].get("metadata", {}).get("sentiment") == "negative")
            
            total = len(model_feedback)
            if total > 0:
                # If mostly negative feedback, reduce temperature for more conservative outputs
                if negative > positive:
                    adjustments["temperature"] = -0.1
                    adjustments["top_p"] = -0.05
                # If mostly positive feedback, can be slightly more creative
                else:
                    adjustments["temperature"] = 0.05
                    adjustments["frequency_penalty"] = 0.05
        
        logger.info(f"Adjusted parameters for {model_type} model: {adjustments}")
        return adjustments
    
    async def get_user_preferences(self) -> Dict[str, Any]:
        """
        Extract user preferences from feedback history.
        
        Returns:
            Dictionary of user preferences
        """
        preferences = {}
        
        # Get user preference insights
        insights = await self.context_engine.get_insights({"type": "feedback_user_preference"})
        
        for insight in insights:
            content = insight["content"]
            # Simple heuristic to extract preferences from feedback
            if "prefer" in content.lower():
                parts = content.split("prefer", 1)
                if len(parts) > 1:
                    preference = parts[1].strip()
                    category = "general"
                    
                    # Try to categorize the preference
                    if any(kw in preference.lower() for kw in ["code", "syntax", "format"]):
                        category = "code_style"
                    elif any(kw in preference.lower() for kw in ["document", "comment"]):
                        category = "documentation"
                    elif any(kw in preference.lower() for kw in ["architecture", "design", "pattern"]):
                        category = "architecture"
                    
                    if category not in preferences:
                        preferences[category] = []
                    
                    preferences[category].append(preference)
        
        return preferences

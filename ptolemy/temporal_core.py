import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy import select, and_

from loguru import logger
from ptolemy.config import TEMPORAL_DIR
from ptolemy.database import async_session, Event


class TemporalCore:
    """
    Temporal Core manages the continuous event-stream storage, replacing linear version control.
    It provides comprehensive project history and decision rationale logging.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or TEMPORAL_DIR
        self.current_events = []
        self.db_enabled = True
    
    async def initialize(self):
        """Initialize the Temporal Core system."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporal Core initialized with storage path: {self.storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Temporal Core: {str(e)}")
            raise
    
    async def _save_event_to_file(self, event: Dict[str, Any]) -> Path:
        """
        Save an event to a file in the storage path.
        
        Args:
            event: The event to save
            
        Returns:
            The path to the saved file
        """
        event_file_path = self.storage_path / f"{event['id']}.json"
        with open(event_file_path, 'w') as f:
            json.dump(event, f, indent=2)
        return event_file_path
    
    async def record_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a new event in the temporal core.
        
        Args:
            event_type: The type of event being recorded
            event_data: The data associated with the event
            
        Returns:
            The recorded event object
        """
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event_data
        }
        
        try:
            # Store in file system for backward compatibility
            await self._save_event_to_file(event)
            
            # Store in database if enabled
            if self.db_enabled:
                async with async_session() as session:
                    db_event = Event(
                        id=event["id"],
                        timestamp=datetime.fromisoformat(event["timestamp"]),
                        type=event_type,
                        data=event_data
                    )
                    session.add(db_event)
                    await session.commit()
            
            self.current_events.append(event)
            logger.info(f"Event recorded: {event_type} ({event['id']})")
            return event
        except Exception as e:
            logger.error(f"Failed to record event: {str(e)}")
            raise
    
    async def get_events(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve events from the temporal core, optionally filtered.
        
        Args:
            filters: Optional filters to apply to the events
            
        Returns:
            A list of events matching the filters
        """
        filters = filters or {}
        
        try:
            if self.db_enabled:
                # Use database query with SQLAlchemy
                async with async_session() as session:
                    query = select(Event)
                    
                    # Apply filters
                    if "type" in filters:
                        query = query.where(Event.type == filters["type"])
                    if "time_after" in filters:
                        query = query.where(Event.timestamp > datetime.fromisoformat(filters["time_after"]))
                    if "time_before" in filters:
                        query = query.where(Event.timestamp < datetime.fromisoformat(filters["time_before"]))
                    if "project_id" in filters:
                        # This assumes project_id is stored in the data JSON
                        # For proper JSON field querying, consider PostgreSQL
                        pass
                    
                    # Order by timestamp
                    query = query.order_by(Event.timestamp)
                    
                    result = await session.execute(query)
                    db_events = result.scalars().all()
                    
                    # Convert to dict format for API consistency
                    events = []
                    for db_event in db_events:
                        events.append({
                            "id": db_event.id,
                            "timestamp": db_event.timestamp.isoformat(),
                            "type": db_event.type,
                            "data": db_event.data
                        })
                    
                    return events
            else:
                # Fall back to file-based retrieval (as in your original code)
                events = []
                for event_file in self.storage_path.glob("*.json"):
                    with open(event_file, 'r') as f:
                        event_data = json.load(f)
                    
                    # Apply filters if any
                    include_event = True
                    for key, value in filters.items():
                        if key == "type" and event_data["type"] != value:
                            include_event = False
                            break
                        if key == "time_after" and datetime.fromisoformat(event_data["timestamp"]) <= datetime.fromisoformat(value):
                            include_event = False
                            break
                        if key == "time_before" and datetime.fromisoformat(event_data["timestamp"]) >= datetime.fromisoformat(value):
                            include_event = False
                            break
                        if key == "project_id" and event_data.get("data", {}).get("project_id") != value:
                            include_event = False
                            break
                    
                    if include_event:
                        events.append(event_data)
                
                # Sort events by timestamp
                return sorted(events, key=lambda x: x["timestamp"])
        except Exception as e:
            logger.error(f"Failed to get events: {str(e)}")
            raise
    
    async def get_event_by_id(self, event_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific event by its ID.
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            The event object
        """
        try:
            if self.db_enabled:
                async with async_session() as session:
                    query = select(Event).where(Event.id == event_id)
                    result = await session.execute(query)
                    db_event = result.scalars().first()
                    
                    if db_event:
                        return {
                            "id": db_event.id,
                            "timestamp": db_event.timestamp.isoformat(),
                            "type": db_event.type,
                            "data": db_event.data
                        }
                    else:
                        raise ValueError(f"Event with ID {event_id} not found")
            else:
                # File-based retrieval
                event_file_path = self.storage_path / f"{event_id}.json"
                with open(event_file_path, 'r') as f:
                    event_data = json.load(f)
                return event_data
        except Exception as e:
            logger.error(f"Failed to get event by ID: {str(e)}")
            raise
    
    async def get_event_stream(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a stream of events between the specified time range.
        
        Args:
            start_time: Optional ISO format start time
            end_time: Optional ISO format end time
            
        Returns:
            A list of events within the time range
        """
        filters = {}
        if start_time:
            filters["time_after"] = start_time
        if end_time:
            filters["time_before"] = end_time
        return await self.get_events(filters)

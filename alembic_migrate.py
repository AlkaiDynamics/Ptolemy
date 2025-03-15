"""
Migration script to move data from JSON files to SQLite database.
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
import os
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ptolemy.database import Base, Event, Relationship, Pattern, Insight
from ptolemy.config import TEMPORAL_DIR, CONTEXT_DIR, BASE_DIR

# Create synchronous engine for migration
DATABASE_PATH = BASE_DIR / "data" / "ptolemy.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)

def init_db():
    """Initialize the database by creating all tables."""
    # Create the database directory if it doesn't exist
    os.makedirs(DATABASE_PATH.parent, exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(engine)
    print("Database schema created successfully")

def migrate_events():
    """Migrate events from JSON files to database"""
    session = Session()
    try:
        for event_file in TEMPORAL_DIR.glob("*.json"):
            with open(event_file, 'r') as f:
                event_data = json.load(f)
                
            # Create db model
            event = Event(
                id=event_data["id"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                type=event_data["type"],
                data=event_data["data"]
            )
            
            session.add(event)
        
        session.commit()
        print(f"Migrated events to database")
    except Exception as e:
        session.rollback()
        print(f"Error migrating events: {e}")
    finally:
        session.close()

def migrate_relationships():
    """Migrate relationships from JSON files to database"""
    relationships_path = CONTEXT_DIR / "relationships"
    if not relationships_path.exists():
        print(f"Relationships directory not found: {relationships_path}")
        return
        
    session = Session()
    try:
        for rel_file in relationships_path.glob("*.json"):
            with open(rel_file, 'r') as f:
                rel_data = json.load(f)
                
            # Create db model
            rel = Relationship(
                id=rel_data["id"],
                source_entity=rel_data["source_entity"],
                target_entity=rel_data["target_entity"],
                relationship_type=rel_data["relationship_type"],
                meta_data=rel_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(rel_data["timestamp"])
            )
            
            session.add(rel)
        
        session.commit()
        print(f"Migrated relationships to database")
    except Exception as e:
        session.rollback()
        print(f"Error migrating relationships: {e}")
    finally:
        session.close()

def migrate_patterns():
    """Migrate patterns from JSON files to database"""
    patterns_path = CONTEXT_DIR / "patterns"
    if not patterns_path.exists():
        print(f"Patterns directory not found: {patterns_path}")
        return
        
    session = Session()
    try:
        for pattern_file in patterns_path.glob("*.json"):
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
                
            # Create db model
            pattern = Pattern(
                id=pattern_data["id"],
                name=pattern_data["name"],
                type=pattern_data["type"],
                implementation=pattern_data["implementation"],
                meta_data=pattern_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(pattern_data["timestamp"])
            )
            
            session.add(pattern)
        
        session.commit()
        print(f"Migrated patterns to database")
    except Exception as e:
        session.rollback()
        print(f"Error migrating patterns: {e}")
    finally:
        session.close()

def migrate_insights():
    """Migrate insights from JSON files to database"""
    insights_path = CONTEXT_DIR / "insights"
    if not insights_path.exists():
        print(f"Insights directory not found: {insights_path}")
        return
        
    session = Session()
    try:
        for insight_file in insights_path.glob("*.json"):
            with open(insight_file, 'r') as f:
                insight_data = json.load(f)
                
            # Create db model
            insight = Insight(
                id=insight_data["id"],
                type=insight_data["type"],
                content=insight_data["content"],
                relevance=insight_data.get("relevance", 0.5),
                meta_data=insight_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(insight_data["timestamp"])
            )
            
            session.add(insight)
        
        session.commit()
        print(f"Migrated insights to database")
    except Exception as e:
        session.rollback()
        print(f"Error migrating insights: {e}")
    finally:
        session.close()

def main():
    # Initialize the database
    init_db()
    
    # Migrate data
    migrate_events()
    migrate_relationships()
    migrate_patterns()
    migrate_insights()
    
    print("Migration completed successfully")

if __name__ == "__main__":
    main()

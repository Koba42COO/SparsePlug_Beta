import os
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class CreditEvent:
    timestamp: str
    event_type: str  # "model_access", "slice_served", "training_run"
    model_id: str
    details: Dict
    attribution: Optional[Dict] = None

class AuditLogger:
    """
    Logs provenance and credit events to a persistent history.
    """
    
    def __init__(self, history_dir: str = "history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.markdown_file = self.history_dir / "HISTORY.md"
        
        # Initialize Markdown if missing
        if not self.markdown_file.exists():
            with open(self.markdown_file, 'w') as f:
                f.write("# Chronicle of Intelligence\n")
                f.write("A continuous history of model usage and credit distribution.\n\n")

    def log_event(self, event_type: str, model_id: str, details: Dict, attribution: Dict = None):
        """Log an event to JSONL and Markdown."""
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")
        
        # 1. Create Event Object
        event = CreditEvent(
            timestamp=timestamp,
            event_type=event_type,
            model_id=model_id,
            details=details,
            attribution=attribution
        )
        
        # 2. Append to JSONL (Partitioned by Date)
        jsonl_file = self.history_dir / f"{date_str}.jsonl"
        with open(jsonl_file, 'a') as f:
            f.write(json.dumps(asdict(event)) + "\n")
            
        # 3. Append to HISTORY.md (Human Readable)
        self._append_to_markdown(event)
        
    def _append_to_markdown(self, event: CreditEvent):
        """Append a human-readable entry to the chronicle."""
        with open(self.markdown_file, 'a') as f:
            f.write(f"## {event.timestamp} - {event.event_type.replace('_', ' ').title()}\n")
            
            if event.model_id:
                f.write(f"**Model**: `{event.model_id}`\n\n")
            
            if event.attribution:
                f.write("**Attribution**:\n")
                for k, v in event.attribution.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")
                
            f.write("**Details**:\n")
            f.write("```json\n")
            f.write(json.dumps(event.details, indent=2))
            f.write("\n```\n")
            f.write("---\n\n")

    def log_external_access(self, tool_name: str, source: str, details: Dict):
        """
        Log usage of an external tool or data source.
        Useful for tracking novelty/credit from scrapes or API calls.
        """
        self.log_event(
            event_type="external_tool_use",
            model_id="system_agent", # It's the system acting, not a specific model file
            details={
                "tool": tool_name, 
                "source": source, 
                **details
            },
            attribution={"source_url": source}
        )

# Global Instance
audit_logger = AuditLogger()

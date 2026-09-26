"""Section 11: the audit trail. Append-only JSON Lines — simple, diffable,
greppable, and trivially replayable, which is what R8 asks for without
needing a database server for a hackathon build."""
import json
import time
from pathlib import Path


class AuditLogger:
    def __init__(self, path: str = "data/audit_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields):
        record = {"event": event_type, "logged_at": time.time(), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def replay(self):
        """Yields every logged record in order — the basis of the 'replay
        tool' mentioned in Section 11."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

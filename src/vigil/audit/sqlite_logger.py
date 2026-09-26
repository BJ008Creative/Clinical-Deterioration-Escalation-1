"""SQLite backend for the audit trail (Section 5's tech-stack table: "SQLite
(upgradeable to PostgreSQL); JSON Lines export of the audit log"). Same
interface as AuditLogger (log/replay) so pipeline.py doesn't care which
backend is wired in — swap one line in the caller. Uses only the stdlib
sqlite3 module, so unlike FastAPI/LangGraph/Streamlit this is fully testable
without network access.
"""
import json
import sqlite3
import time
from pathlib import Path


class SQLiteAuditLogger:
    def __init__(self, path: str = "data/audit.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                patient_id TEXT,
                timestamp REAL,
                logged_at REAL NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_patient ON audit_log(patient_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event ON audit_log(event)")
        self._conn.commit()

    def log(self, event_type: str, **fields):
        record = {"event": event_type, "logged_at": time.time(), **fields}
        self._conn.execute(
            "INSERT INTO audit_log (event, patient_id, timestamp, logged_at, payload) VALUES (?, ?, ?, ?, ?)",
            (event_type, fields.get("patient_id"), fields.get("timestamp"),
             record["logged_at"], json.dumps(record, default=str)),
        )
        self._conn.commit()
        return record

    def replay(self, patient_id: str = None, event: str = None):
        """Yields records in insertion order, optionally filtered — the
        SQL-backed equivalent of AuditLogger.replay()."""
        query = "SELECT payload FROM audit_log WHERE 1=1"
        params = []
        if patient_id is not None:
            query += " AND patient_id = ?"
            params.append(patient_id)
        if event is not None:
            query += " AND event = ?"
            params.append(event)
        query += " ORDER BY id ASC"
        for (payload,) in self._conn.execute(query, params):
            yield json.loads(payload)

    def export_jsonl(self, out_path: str):
        """Section 5: 'JSON Lines export of the audit log' — the SQLite
        store is the primary backend, JSONL is a derived export."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for record in self.replay():
                f.write(json.dumps(record, default=str) + "\n")

    def close(self):
        self._conn.close()

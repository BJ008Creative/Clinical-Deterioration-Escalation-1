"""Section 14.1 names 'state/ # patient state manager, repositories'
explicitly. This is that repository: periodic SQLite snapshots of each
patient's derived state (not the full in-memory object — the live
PatientState with its deque/enum fields stays in memory for speed; this is
what a restart or another service would read to recover cohort state).
stdlib sqlite3 only, so fully testable here.
"""
import json
import sqlite3
from pathlib import Path


class SQLitePatientRepository:
    def __init__(self, path: str = "data/state.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patient_snapshots (
                patient_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                physiological_state TEXT NOT NULL,
                alert_state TEXT NOT NULL,
                evidence REAL NOT NULL,
                features TEXT NOT NULL,
                profile TEXT NOT NULL,
                PRIMARY KEY (patient_id, timestamp)
            )
        """)
        self._conn.commit()

    def save_snapshot(self, patient_state, timestamp: float):
        self._conn.execute(
            """INSERT OR REPLACE INTO patient_snapshots
               (patient_id, timestamp, physiological_state, alert_state, evidence, features, profile)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (patient_state.patient_id, timestamp, patient_state.physio_state.value,
             patient_state.alert_state.value, patient_state.evidence,
             json.dumps(patient_state.last_features, default=str),
             json.dumps(patient_state.profile, default=str)),
        )
        self._conn.commit()

    def load_latest(self, patient_id: str):
        row = self._conn.execute(
            """SELECT patient_id, timestamp, physiological_state, alert_state, evidence, features, profile
               FROM patient_snapshots WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 1""",
            (patient_id,),
        ).fetchone()
        if row is None:
            return None
        pid, ts, physio, alert_state, evidence, features, profile = row
        return {"patient_id": pid, "timestamp": ts, "physiological_state": physio,
                "alert_state": alert_state, "evidence": evidence,
                "features": json.loads(features), "profile": json.loads(profile)}

    def load_all_latest(self) -> list:
        """One row per patient — the basis of a cohort worklist rebuilt from
        disk after a restart."""
        rows = self._conn.execute("""
            SELECT s.patient_id, s.timestamp, s.physiological_state, s.alert_state, s.evidence, s.features, s.profile
            FROM patient_snapshots s
            INNER JOIN (
                SELECT patient_id, MAX(timestamp) AS max_ts FROM patient_snapshots GROUP BY patient_id
            ) latest ON s.patient_id = latest.patient_id AND s.timestamp = latest.max_ts
        """).fetchall()
        return [{"patient_id": r[0], "timestamp": r[1], "physiological_state": r[2],
                 "alert_state": r[3], "evidence": r[4], "features": json.loads(r[5]),
                 "profile": json.loads(r[6])} for r in rows]

    def close(self):
        self._conn.close()

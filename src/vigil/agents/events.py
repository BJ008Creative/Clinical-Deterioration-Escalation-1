"""Section 5.2's Observer/Pub-Sub pattern: 'New readings, state changes,
alerts and clinician actions are published as events consumed by the UI and
audit logger.' Previously pipeline.py called audit_logger.log(...) directly,
which works but isn't actually the pattern the report names — the pipeline
knew about the audit logger specifically. This decouples that: pipeline.py
publishes events, and anything (the audit logger, the dashboard, a future
metrics collector) can subscribe without pipeline.py knowing they exist.
"""
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._subscribers = defaultdict(list)

    def subscribe(self, event_type: str, handler):
        """handler: callable(event_type: str, **fields) -> None"""
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler):
        """Convenience: subscribe to every event type ('*')."""
        self._subscribers["*"].append(handler)

    def publish(self, event_type: str, **fields):
        for handler in self._subscribers.get(event_type, []):
            handler(event_type, **fields)
        for handler in self._subscribers.get("*", []):
            handler(event_type, **fields)


def audit_subscriber(audit_logger):
    """Returns a handler that forwards every published event straight to an
    AuditLogger/SQLiteAuditLogger — the audit logger becomes just one
    observer among possibly several, not something pipeline.py calls by name."""
    def handler(event_type: str, **fields):
        audit_logger.log(event_type, **fields)
    return handler

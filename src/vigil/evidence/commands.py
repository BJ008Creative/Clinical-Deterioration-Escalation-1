"""Section 5.2's Command pattern: 'Clinician actions are command objects
that are applied to state and logged.' apply_clinician_action() previously
did this as a single dispatch function — functionally the same outcome, but
not literally command objects, and with no way to undo a misclick. This
module is the real pattern: each command captures enough of the prior state
to be undone, and command_history on PatientState is the audit trail of
which commands ran.
"""
from dataclasses import dataclass, field

from vigil.config import DELTA_DISMISS_EXTRA, DEFER_MINUTES
from vigil.models import AlertState, news2_risk_band


class ClinicianCommand:
    """Base class. Concrete commands implement execute() and undo()."""
    name = "base"

    def __init__(self, patient_state, now: float, reason: str = ""):
        self.patient_state = patient_state
        self.now = now
        self.reason = reason
        self._snapshot = None  # set by execute(), used by undo()

    def execute(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError

    def _record(self):
        entry = {"command": self.name, "reason": self.reason, "at": self.now}
        self.patient_state.command_history.append(entry)
        if self.patient_state.last_alert is not None:
            action_record = {"action": self.name, "reason": self.reason, "at": self.now}
            self.patient_state.last_alert["clinician_action"] = action_record
            if self.patient_state.alert_history:
                self.patient_state.alert_history[-1]["clinician_action"] = action_record


class AcceptCommand(ClinicianCommand):
    name = "accept"

    def execute(self):
        self._snapshot = {"alert_state": self.patient_state.alert_state}
        self.patient_state.alert_state = AlertState.ACKNOWLEDGED
        self._record()

    def undo(self):
        self.patient_state.alert_state = self._snapshot["alert_state"]


class DismissCommand(ClinicianCommand):
    name = "dismiss"

    def execute(self):
        self._snapshot = {"alert_state": self.patient_state.alert_state,
                           "dismiss_bar": self.patient_state.dismiss_bar}
        self.patient_state.alert_state = AlertState.DISMISSED
        self.patient_state.dismiss_bar += DELTA_DISMISS_EXTRA
        self._record()

    def undo(self):
        self.patient_state.alert_state = self._snapshot["alert_state"]
        self.patient_state.dismiss_bar = self._snapshot["dismiss_bar"]


class DeferCommand(ClinicianCommand):
    name = "defer"

    def execute(self):
        self._snapshot = {"alert_state": self.patient_state.alert_state,
                           "defer_until": self.patient_state.defer_until,
                           "defer_band_state": self.patient_state.defer_band_state,
                           "defer_news2_band": self.patient_state.defer_news2_band}
        self.patient_state.alert_state = AlertState.DEFERRED
        self.patient_state.defer_until = self.now + DEFER_MINUTES * 60
        self.patient_state.defer_band_state = self.patient_state.physio_state
        self.patient_state.defer_news2_band = news2_risk_band(
            self.patient_state.last_features.get("partial_news2", 0))
        self._record()

    def undo(self):
        self.patient_state.alert_state = self._snapshot["alert_state"]
        self.patient_state.defer_until = self._snapshot["defer_until"]
        self.patient_state.defer_band_state = self._snapshot["defer_band_state"]
        self.patient_state.defer_news2_band = self._snapshot["defer_news2_band"]


class InvestigateCommand(ClinicianCommand):
    name = "investigate"

    def execute(self):
        self._snapshot = {"alert_state": self.patient_state.alert_state,
                           "investigation_mode": self.patient_state.investigation_mode}
        self.patient_state.alert_state = AlertState.OPEN
        self.patient_state.investigation_mode = True
        self._record()

    def undo(self):
        self.patient_state.alert_state = self._snapshot["alert_state"]
        self.patient_state.investigation_mode = self._snapshot["investigation_mode"]


COMMANDS = {"accept": AcceptCommand, "dismiss": DismissCommand,
            "defer": DeferCommand, "investigate": InvestigateCommand}


def run_command(patient_state, action: str, now: float, reason: str = "", audit_logger=None, event_bus=None):
    """Constructs, executes, and returns the command object (so the caller
    can hold onto it for undo())."""
    if action not in COMMANDS:
        raise ValueError(f"unknown clinician action: {action}")
    command = COMMANDS[action](patient_state, now, reason)
    command.execute()
    if event_bus is not None:
        event_bus.publish("clinician_action", patient_id=patient_state.patient_id, timestamp=now,
                           action=action, reason=reason, resulting_alert_state=patient_state.alert_state.value)
    elif audit_logger is not None:
        audit_logger.log("clinician_action", patient_id=patient_state.patient_id, timestamp=now,
                          action=action, reason=reason, resulting_alert_state=patient_state.alert_state.value)
    return command

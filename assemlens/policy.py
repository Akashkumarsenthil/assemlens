"""Deterministic routing and verification; no model or cloud calls here."""
from dataclasses import dataclass, field
from enum import Enum

class Route(str, Enum):
    LOCAL = 'local'
    RECAPTURE = 'recapture'
    CLOUD = 'cloud'
    ABSTAIN = 'abstain'

@dataclass(frozen=True)
class Evidence:
    supported_product: bool
    clear_view: bool
    unresolved_observations: int = 0
    cloud_permission: bool = False
    cloud_available: bool = False
    cloud_requests_remaining: int = 0

@dataclass(frozen=True)
class Decision:
    route: Route
    reason: str

def route(e: Evidence) -> Decision:
    if e.unresolved_observations < 0 or e.cloud_requests_remaining < 0:
        raise ValueError('Counts must be nonnegative')
    if not e.clear_view:
        return Decision(Route.RECAPTURE, 'Insufficient visual evidence: ask for another angle')
    if not e.supported_product:
        return Decision(Route.ABSTAIN, 'No verified product package: do not guide unsupported assembly')
    if e.unresolved_observations < 2:
        return Decision(Route.LOCAL, 'Evaluate supported step on the local device')
    if e.cloud_permission and e.cloud_available and e.cloud_requests_remaining > 0:
        return Decision(Route.CLOUD, 'Repeated unresolved clear observations; escalation permitted')
    return Decision(Route.ABSTAIN, 'Escalation unavailable or not permitted; preserve current step')

@dataclass
class StepVerifier:
    """Require two distinct, consecutive complete captures of the current step.

    The caller must hash actual image bytes for capture_id, not fabricate IDs.
    This gates predictions; it does not establish their visual accuracy.
    """
    step_id: str
    required: int = 2
    _complete: set[str] = field(default_factory=set, init=False)
    _seen: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if not self.step_id or self.required < 2:
            raise ValueError('A step and at least two captures are required')

    @property
    def ready(self): return len(self._complete) >= self.required

    def observe(self, step_id: str, capture_id: str, verdict: str):
        if step_id != self.step_id: raise ValueError('Observation is for another step')
        if verdict not in {'complete','incomplete','incorrect','uncertain'}:
            raise ValueError('Unknown verdict')
        if not capture_id: raise ValueError('Capture identity is required')
        if capture_id in self._seen: return self.ready
        self._seen.add(capture_id)
        if verdict == 'complete': self._complete.add(capture_id)
        else: self._complete.clear()
        return self.ready

    def advance(self, next_step_id: str):
        if not self.ready: raise ValueError('Step is not verified')
        if not next_step_id or next_step_id == self.step_id:
            raise ValueError('A different next step is required')
        self.step_id=next_step_id; self._complete.clear(); self._seen.clear()

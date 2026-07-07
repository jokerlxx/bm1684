from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Hashable, Iterable, Tuple


NORMAL = "NORMAL"
SUSPECTING = "SUSPECTING"
CONFIRMED = "CONFIRMED"
ALERTING = "ALERTING"
COOLDOWN = "COOLDOWN"


def timestamp_to_seconds(timestamp: Any, fallback: float | None = None) -> float:
    if isinstance(timestamp, datetime):
        return float(timestamp.timestamp())
    try:
        return float(timestamp)
    except Exception:
        return time.time() if fallback is None else float(fallback)


@dataclass
class EventDecision:
    event_id: Hashable
    state: str
    previous_state: str
    hit_ratio: float
    observation_span: float
    observation_count: int
    hit_count: int
    just_confirmed: bool = False
    just_alerted: bool = False
    cooldown_remaining_s: float = 0.0

    @property
    def has_recent_hit(self) -> bool:
        return self.hit_count > 0

    @property
    def is_confirmed(self) -> bool:
        return self.state in {CONFIRMED, ALERTING, COOLDOWN}

    @property
    def should_display(self) -> bool:
        return self.is_confirmed and self.has_recent_hit


class TrackEvent:
    def __init__(
        self,
        event_id: Hashable,
        window_seconds: float,
        threshold_ratio: float,
        min_observation_seconds: float,
        cooldown_seconds: float,
        alert_hold_seconds: float,
    ):
        self.event_id = event_id
        self.window_seconds = max(0.1, float(window_seconds))
        self.threshold_ratio = max(0.0, min(1.0, float(threshold_ratio)))
        self.min_observation_seconds = max(0.0, float(min_observation_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.alert_hold_seconds = max(0.0, float(alert_hold_seconds))
        self.samples: Deque[Tuple[float, bool]] = deque()
        self.state = NORMAL
        self.last_seen = 0.0
        self.alerting_until = 0.0
        self.cooldown_until = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def _stats(self, now: float) -> tuple[float, float, int, int]:
        self._prune(now)
        observation_count = len(self.samples)
        hit_count = sum(1 for _, hit in self.samples if hit)
        hit_ratio = (hit_count / observation_count) if observation_count else 0.0
        if observation_count > 1:
            observation_span = max(0.0, self.samples[-1][0] - self.samples[0][0])
        else:
            observation_span = 0.0
        return hit_ratio, observation_span, observation_count, hit_count

    def _advance_timers(self, now: float) -> None:
        if self.state == ALERTING and now >= self.alerting_until:
            self.state = COOLDOWN
        if self.state == COOLDOWN and now >= self.cooldown_until:
            self.state = NORMAL

    def update(self, now: float, hit: bool, valid: bool = True) -> EventDecision:
        now = float(now)
        previous_state = self.state
        self.last_seen = now
        self._advance_timers(now)

        if valid:
            self.samples.append((now, bool(hit)))

        hit_ratio, observation_span, observation_count, hit_count = self._stats(now)
        can_confirm = (
            observation_count > 0
            and observation_span >= self.min_observation_seconds
            and hit_ratio >= self.threshold_ratio
        )
        just_confirmed = False
        just_alerted = False

        if self.state == NORMAL:
            if hit_count > 0:
                self.state = SUSPECTING

        if self.state == SUSPECTING:
            if can_confirm:
                self.state = CONFIRMED
                just_confirmed = True
            elif hit_count == 0:
                self.state = NORMAL

        if self.state == CONFIRMED:
            self.state = ALERTING
            self.alerting_until = now + self.alert_hold_seconds
            self.cooldown_until = now + self.cooldown_seconds
            just_alerted = True

        cooldown_remaining = (
            max(0.0, self.cooldown_until - now)
            if self.state in {ALERTING, COOLDOWN}
            else 0.0
        )
        return EventDecision(
            event_id=self.event_id,
            state=self.state,
            previous_state=previous_state,
            hit_ratio=hit_ratio,
            observation_span=observation_span,
            observation_count=observation_count,
            hit_count=hit_count,
            just_confirmed=just_confirmed,
            just_alerted=just_alerted,
            cooldown_remaining_s=cooldown_remaining,
        )

    def peek(self, now: float) -> EventDecision:
        now = float(now)
        previous_state = self.state
        self._advance_timers(now)
        hit_ratio, observation_span, observation_count, hit_count = self._stats(now)
        cooldown_remaining = (
            max(0.0, self.cooldown_until - now)
            if self.state in {ALERTING, COOLDOWN}
            else 0.0
        )
        return EventDecision(
            event_id=self.event_id,
            state=self.state,
            previous_state=previous_state,
            hit_ratio=hit_ratio,
            observation_span=observation_span,
            observation_count=observation_count,
            hit_count=hit_count,
            cooldown_remaining_s=cooldown_remaining,
        )


class TrackEventStateMachine:
    def __init__(
        self,
        window_seconds: float,
        threshold_ratio: float,
        min_observation_seconds: float,
        cooldown_seconds: float,
        alert_hold_seconds: float = 0.2,
    ):
        self.window_seconds = float(window_seconds)
        self.threshold_ratio = float(threshold_ratio)
        self.min_observation_seconds = float(min_observation_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.alert_hold_seconds = float(alert_hold_seconds)
        self.events: Dict[Hashable, TrackEvent] = {}

    def get(self, event_id: Hashable) -> TrackEvent:
        if event_id not in self.events:
            self.events[event_id] = TrackEvent(
                event_id=event_id,
                window_seconds=self.window_seconds,
                threshold_ratio=self.threshold_ratio,
                min_observation_seconds=self.min_observation_seconds,
                cooldown_seconds=self.cooldown_seconds,
                alert_hold_seconds=self.alert_hold_seconds,
            )
        return self.events[event_id]

    def update(self, event_id: Hashable, now: float, hit: bool, valid: bool = True) -> EventDecision:
        return self.get(event_id).update(now, hit=hit, valid=valid)

    def peek(self, event_id: Hashable, now: float) -> EventDecision:
        return self.get(event_id).peek(now)

    def decisions(self, now: float) -> Dict[Hashable, EventDecision]:
        return {event_id: event.peek(now) for event_id, event in list(self.events.items())}

    def cleanup(self, active_ids: Iterable[Hashable], now: float, stale_seconds: float) -> None:
        active = set(active_ids)
        max_age = max(float(stale_seconds), self.window_seconds)
        for event_id, event in list(self.events.items()):
            if event_id in active:
                continue
            if now - event.last_seen > max_age:
                self.events.pop(event_id, None)

    def active_count(self, now: float) -> int:
        return sum(1 for decision in self.decisions(now).values() if decision.state in {ALERTING, COOLDOWN})

    def max_cooldown_remaining(self, now: float) -> float:
        values = [decision.cooldown_remaining_s for decision in self.decisions(now).values()]
        return max(values) if values else 0.0

"""Arbitration as inspectable code, not model judgment.

A policy chooses among verdicts. It never chooses whether evidence was required --
the ``no surviving evidence -> unknown`` short-circuit lives above it, in
``FactChecker``, so no policy can be written that answers on zero evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..config import AggregationConfig
from ..errors import ConfigError
from ..ports import AggregationPolicy
from ..types import Evidence, Stance, Verdict


class ContradictionWins:
    """Any *contradicts* wins; else enough *entails* is supported; else unknown.

    Contradiction outranks support because a corpus that both asserts and refutes a
    claim has not established it. The conflict is surfaced as a flag by the caller
    rather than resolved here, so a superseded finding is reported rather than
    silently overwritten.
    """

    name = "contradiction_wins"

    def __init__(self, cfg: AggregationConfig) -> None:
        self._min_supporting = cfg.min_supporting_quotes

    def decide(self, evidence: Sequence[Evidence]) -> Verdict:
        stances = [e.stance for e in evidence]
        if Stance.CONTRADICTS in stances:
            return Verdict.CONTRADICTED
        if stances.count(Stance.ENTAILS) >= self._min_supporting:
            return Verdict.SUPPORTED
        return Verdict.UNKNOWN


class MajorityStance:
    """The obvious alternative, implemented so ``contradiction_wins`` can earn its row.

    A port with one implementation is an assertion; a port with two is a measurement.
    This one is the natural rival: it treats a single refuting quote among many
    supporting ones as noise rather than as a reason to withhold an answer.
    """

    name = "majority_stance"

    def __init__(self, cfg: AggregationConfig) -> None:
        self._min_supporting = cfg.min_supporting_quotes

    def decide(self, evidence: Sequence[Evidence]) -> Verdict:
        entails = sum(1 for e in evidence if e.stance is Stance.ENTAILS)
        contradicts = sum(1 for e in evidence if e.stance is Stance.CONTRADICTS)
        if contradicts > entails:
            return Verdict.CONTRADICTED
        if entails > contradicts and entails >= self._min_supporting:
            return Verdict.SUPPORTED
        return Verdict.UNKNOWN


_POLICIES: dict[str, Callable[[AggregationConfig], AggregationPolicy]] = {
    ContradictionWins.name: ContradictionWins,
    MajorityStance.name: MajorityStance,
}


def get_policy(cfg: AggregationConfig) -> AggregationPolicy:
    try:
        return _POLICIES[cfg.policy](cfg)
    except KeyError:
        raise ConfigError(
            f"unknown aggregation policy '{cfg.policy}'. Known: {', '.join(sorted(_POLICIES))}"
        ) from None

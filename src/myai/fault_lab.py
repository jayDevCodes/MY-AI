from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FaultCase:
    name: str
    description: str
    inject: Callable[[], None]
    recover: Callable[[], None]
    verify: Callable[[], bool]


@dataclass(frozen=True)
class FaultResult:
    name: str
    detected: bool
    recovered: bool
    verified: bool


class FaultInjectionLab:
    """Controlled, opt-in fault tests for the self-healing subsystem."""

    def run(self, case: FaultCase, *, detector: Callable[[], bool]) -> FaultResult:
        detected = False
        recovered = False
        try:
            case.inject()
            detected = bool(detector())
            if detected:
                case.recover()
                recovered = True
            verified = bool(case.verify())
        finally:
            # Fault cases own their rollback/cleanup through recover().
            if not recovered:
                try:
                    case.recover()
                except Exception:
                    pass
        return FaultResult(case.name, detected, recovered, verified)

    @staticmethod
    def simple_toggle(state: dict[str, bool], key: str) -> FaultCase:
        def inject() -> None:
            state[key] = False

        def recover() -> None:
            state[key] = True

        def verify() -> bool:
            return state.get(key) is True

        return FaultCase(
            name=f"toggle:{key}",
            description=f"temporarily disable invariant {key}",
            inject=inject,
            recover=recover,
            verify=verify,
        )

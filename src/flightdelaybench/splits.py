"""Study split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceRole(StrEnum):
    WARMUP = "warmup"
    DEVELOPMENT_TRAIN = "development_train"
    ROLLING_EVALUATION = "rolling_evaluation"
    SELECTION_CALIBRATION = "selection_calibration"
    RETROSPECTIVE_AUDIT = "retrospective_audit"
    LOCKED_CONFIRMATION = "locked_confirmation"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    warmup_year: int = 2010
    minimum_training_year: int = 2011
    rolling_evaluation_years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
    selection_year: int = 2024
    retrospective_year: int = 2025
    confirmation_year: int = 2026
    confirmation_months: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def role_for(self, year: int, month: int | None = None) -> EvidenceRole:
        """Return the prespecified evidence role for a calendar period."""

        if year == self.warmup_year:
            return EvidenceRole.WARMUP
        if self.minimum_training_year <= year < self.rolling_evaluation_years[0]:
            return EvidenceRole.DEVELOPMENT_TRAIN
        if year in self.rolling_evaluation_years:
            return EvidenceRole.ROLLING_EVALUATION
        if year == self.selection_year:
            return EvidenceRole.SELECTION_CALIBRATION
        if year == self.retrospective_year:
            return EvidenceRole.RETROSPECTIVE_AUDIT
        if year == self.confirmation_year and month in self.confirmation_months:
            return EvidenceRole.LOCKED_CONFIRMATION
        return EvidenceRole.OUT_OF_SCOPE

    def training_years_for(self, evaluation_year: int) -> tuple[int, ...]:
        """Return strictly earlier model-training years for a rolling evaluation."""

        if evaluation_year <= self.minimum_training_year:
            raise ValueError("evaluation year must follow the minimum training year")
        if evaluation_year > self.selection_year:
            raise ValueError("post-selection years require an explicitly locked operational fit")
        return tuple(range(self.minimum_training_year, evaluation_year))


DEFAULT_PROTOCOL = StudyProtocol()

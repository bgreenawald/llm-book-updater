"""Phase implementations for book text transformation."""

from book_updater.phases.annotation import IntroductionAnnotationPhase, SummaryAnnotationPhase
from book_updater.phases.base import LlmPhase
from book_updater.phases.factory import PhaseFactory
from book_updater.phases.protocol import Phase
from book_updater.phases.standard import StandardLlmPhase
from book_updater.phases.two_stage import StageConfig, TwoStageFinalPhase

__all__ = [
    "Phase",
    "LlmPhase",
    "StandardLlmPhase",
    "IntroductionAnnotationPhase",
    "SummaryAnnotationPhase",
    "TwoStageFinalPhase",
    "StageConfig",
    "PhaseFactory",
]

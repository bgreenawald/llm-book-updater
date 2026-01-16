"""Book Updater - LLM-powered book text transformation and modernization."""

__version__ = "0.1.0"

from book_updater.config import (
    PhaseConfig,
    PhaseType,
    PostProcessorType,
    RunConfig,
    TwoStageModelConfig,
)
from book_updater.pipeline import Pipeline
from book_updater.study_guide import StudyGuideConfig, run_study_guide

__all__ = [
    "__version__",
    "Pipeline",
    "PhaseConfig",
    "PhaseType",
    "PostProcessorType",
    "RunConfig",
    "TwoStageModelConfig",
    "StudyGuideConfig",
    "run_study_guide",
]

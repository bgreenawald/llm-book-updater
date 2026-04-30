"""Book Updater - LLM-powered book text transformation and modernization."""

__version__ = "0.1.0"

from book_updater.abridge_config import create_abridge_run_config, default_abridge_phases
from book_updater.config import (
    AbridgeWriteModelConfig,
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
    "AbridgeWriteModelConfig",
    "TwoStageModelConfig",
    "StudyGuideConfig",
    "run_study_guide",
    "create_abridge_run_config",
    "default_abridge_phases",
]

# Pydantic Migration Plan

## Executive Summary

This document outlines the migration strategy for adopting Pydantic v2 in the LLM Book Updater codebase. The migration will reduce boilerplate validation code, improve type safety, and enhance API integration reliability.

**Expected Benefits:**
- Eliminate 100+ lines of manual validation code
- Type-safe API models with automatic validation
- Centralized environment configuration
- Better developer experience with IDE support
- Consistent error handling across the codebase

**Estimated Impact:**
- Code reduction: ~200-300 lines
- Files affected: ~15 core files
- Risk level: Low (backward compatible migration possible)

---

## Phase 1: Setup & Foundation (Week 1)

### 1.1 Dependency Installation

**Action:** Add Pydantic to project dependencies

```bash
uv pip install "pydantic>=2.0.0" "pydantic-settings>=2.0.0"
```

**Files to modify:**
- `pyproject.toml` - Add pydantic and pydantic-settings to dependencies

**Validation:**
```bash
uv pip list | grep pydantic
python -c "import pydantic; print(pydantic.VERSION)"
```

### 1.2 Create Pydantic Base Configuration

**Action:** Create a new module for shared Pydantic configurations

**New file:** `src/pydantic_config.py`

```python
"""Shared Pydantic configuration and base models."""
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration for all Pydantic models in the project."""

    model_config = ConfigDict(
        # Validate on assignment for immediate feedback
        validate_assignment=True,
        # Allow arbitrary types for compatibility with existing code
        arbitrary_types_allowed=True,
        # Use enum values directly
        use_enum_values=False,
        # Strict mode for better validation
        strict=False,  # Start permissive, tighten later
        # Validate default values
        validate_default=True,
    )
```

---

## Phase 2: Configuration Models (Week 1-2)

**Priority:** HIGH - Eliminates the most boilerplate code

### 2.1 Migrate PhaseConfig

**File:** `src/config.py:143-234`

**Current state:** ~100 lines of manual validation in `__post_init__`

**Migration steps:**

1. Create new Pydantic version alongside existing dataclass
2. Add comprehensive validators
3. Update tests to use new model
4. Replace all usages
5. Remove old dataclass

**New implementation:**

```python
from pydantic import Field, field_validator, model_validator
from typing import Annotated
from .pydantic_config import BaseConfig


class PhaseConfig(BaseConfig):
    """Configuration for a single LLM processing phase."""

    phase_type: PhaseType
    reasoning: dict[str, str] | None = None
    use_batch: bool
    batch_size: Annotated[int, Field(gt=0)] | None = None
    model_config_name: str | None = None
    two_stage_model_config: 'TwoStageModelConfig' | None = None
    post_processors: list[str] | None = None

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        if v is not None:
            for k, val in v.items():
                if not isinstance(k, str) or not isinstance(val, str):
                    raise ValueError("reasoning must be dict[str, str]")
        return v

    @model_validator(mode='after')
    def validate_model_config(self):
        # Cross-field validation
        if self.model_config_name is None and self.two_stage_model_config is None:
            raise ValueError("Must specify either model_config_name or two_stage_model_config")
        if self.model_config_name is not None and self.two_stage_model_config is not None:
            raise ValueError("Cannot specify both model_config_name and two_stage_model_config")
        return self
```

**Files to update:**
- `src/config.py` - Replace PhaseConfig dataclass
- `tests/test_configuration.py` - Update tests
- All files importing PhaseConfig

**Testing checklist:**
- [ ] All existing tests pass
- [ ] Validation errors match previous behavior
- [ ] JSON serialization/deserialization works
- [ ] Type checking passes (mypy/ruff)

### 2.2 Migrate ModelConfig

**File:** `src/llm_model.py`

**Migration:**

```python
class ModelConfig(BaseConfig):
    """Configuration for LLM model parameters."""

    model: str
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @field_validator('model')
    @classmethod
    def validate_model_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("model must be a non-empty string")
        return v
```

### 2.3 Migrate TwoStageModelConfig

**File:** `src/config.py`

```python
class TwoStageModelConfig(BaseConfig):
    """Configuration for two-stage model processing."""

    first_pass_model_config: ModelConfig
    second_pass_model_config: ModelConfig

    @model_validator(mode='after')
    def validate_configs(self):
        if self.first_pass_model_config == self.second_pass_model_config:
            raise ValueError("Two-stage configs should differ")
        return self
```

### 2.4 Migrate RunConfig

**File:** `src/config.py`

```python
from pathlib import Path


class RunConfig(BaseConfig):
    """Configuration for a complete pipeline run."""

    book_name: str
    phases: list[PhaseConfig]
    input_file: Path
    output_file: Path
    metadata_file: Path | None = None

    @field_validator('input_file', 'output_file')
    @classmethod
    def validate_file_paths(cls, v):
        path = Path(v)
        # Validation logic here
        return path

    @model_validator(mode='after')
    def validate_phases(self):
        if not self.phases:
            raise ValueError("Must specify at least one phase")
        return self
```

**Estimated savings:** ~150 lines of validation code eliminated

---

## Phase 3: Cost Tracking Models (Week 2)

**Priority:** HIGH - Improves API integration reliability

### 3.1 Migrate Cost Tracking Dataclasses

**File:** `src/cost_tracker.py`

**Models to migrate:**
- `GenerationStats`
- `PhaseCosts`
- `RunCosts`

**Example migration:**

```python
from pydantic import Field
from decimal import Decimal


class GenerationStats(BaseConfig):
    """Statistics for a single LLM generation."""

    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    total_tokens: Annotated[int, Field(ge=0)] = 0
    cost: Decimal = Decimal('0.00')
    model: str
    timestamp: str

    @model_validator(mode='after')
    def validate_token_totals(self):
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input + output tokens")
        return self


class PhaseCosts(BaseConfig):
    """Cost tracking for a pipeline phase."""

    phase_name: str
    generations: list[GenerationStats] = Field(default_factory=list)
    total_cost: Decimal = Decimal('0.00')
    total_input_tokens: Annotated[int, Field(ge=0)] = 0
    total_output_tokens: Annotated[int, Field(ge=0)] = 0

    def add_generation(self, stats: GenerationStats) -> None:
        """Add a generation and update totals."""
        self.generations.append(stats)
        self.total_cost += stats.cost
        self.total_input_tokens += stats.input_tokens
        self.total_output_tokens += stats.output_tokens


class RunCosts(BaseConfig):
    """Cost tracking for entire pipeline run."""

    book_name: str
    phases: list[PhaseCosts] = Field(default_factory=list)
    total_cost: Decimal = Decimal('0.00')
    started_at: str
    completed_at: str | None = None
```

**Benefits:**
- Automatic validation of token counts
- Type-safe cost calculations
- JSON serialization for reporting

---

## Phase 4: API Response Models (Week 3)

**Priority:** MEDIUM-HIGH - Improves reliability and type safety

### 4.1 Create Provider Response Models

**New file:** `src/api_models.py`

```python
"""Pydantic models for LLM provider API responses."""
from pydantic import Field
from .pydantic_config import BaseConfig


# OpenRouter Models
class OpenRouterUsage(BaseConfig):
    """Token usage from OpenRouter API."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenRouterChoice(BaseConfig):
    """Response choice from OpenRouter."""
    index: int
    message: dict[str, str]
    finish_reason: str | None = None


class OpenRouterResponse(BaseConfig):
    """Complete OpenRouter API response."""
    id: str
    model: str
    choices: list[OpenRouterChoice]
    usage: OpenRouterUsage
    created: int | None = None


# OpenAI Models
class OpenAIUsage(BaseConfig):
    """Token usage from OpenAI API."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIMessage(BaseConfig):
    """Message from OpenAI response."""
    role: str
    content: str | None = None


class OpenAIChoice(BaseConfig):
    """Response choice from OpenAI."""
    index: int
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIResponse(BaseConfig):
    """Complete OpenAI API response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


# Gemini Models
class GeminiUsageMetadata(BaseConfig):
    """Usage metadata from Gemini API."""
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


class GeminiContent(BaseConfig):
    """Content from Gemini response."""
    parts: list[dict]
    role: str


class GeminiCandidate(BaseConfig):
    """Candidate response from Gemini."""
    content: GeminiContent
    finish_reason: str | None = None
    index: int


class GeminiResponse(BaseConfig):
    """Complete Gemini API response."""
    candidates: list[GeminiCandidate]
    usage_metadata: GeminiUsageMetadata
    model_version: str | None = None


# Batch API Models
class BatchRequest(BaseConfig):
    """Generic batch request format."""
    custom_id: str
    method: str = "POST"
    url: str
    body: dict


class BatchResponse(BaseConfig):
    """Generic batch response format."""
    id: str
    custom_id: str
    response: dict | None = None
    error: dict | None = None
```

### 4.2 Update LLM Model to Use Response Models

**File:** `src/llm_model.py`

**Changes:**
- Replace manual dict parsing with Pydantic model validation
- Use `model_validate()` or `model_validate_json()` for API responses
- Type hints updated to use specific response models

**Example:**

```python
def _parse_openrouter_response(self, response_data: dict) -> str:
    """Parse OpenRouter API response with validation."""
    # Old: Manual dict access with no validation
    # content = response_data['choices'][0]['message']['content']

    # New: Validated response model
    response = OpenRouterResponse.model_validate(response_data)
    return response.choices[0].message['content']
```

**Benefits:**
- Automatic validation of API responses
- Clear error messages when API format changes
- Type safety for response handling
- Self-documenting API contracts

---

## Phase 5: Environment Configuration (Week 3)

**Priority:** MEDIUM - Improves configuration management

### 5.1 Create Settings Model

**New file:** `src/settings.py`

```python
"""Application settings using Pydantic Settings."""
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # API Keys
    openrouter_api_key: SecretStr | None = Field(None, alias='OPENROUTER_API_KEY')
    openai_api_key: SecretStr | None = Field(None, alias='OPENAI_API_KEY')
    anthropic_api_key: SecretStr | None = Field(None, alias='ANTHROPIC_API_KEY')
    google_api_key: SecretStr | None = Field(None, alias='GOOGLE_API_KEY')

    # API Configuration
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"

    # Application Settings
    debug: bool = False
    log_level: str = "INFO"

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider."""
        key_map = {
            'openrouter': self.openrouter_api_key,
            'openai': self.openai_api_key,
            'anthropic': self.anthropic_api_key,
            'google': self.google_api_key,
        }
        key = key_map.get(provider.lower())
        return key.get_secret_value() if key else None


# Global settings instance
settings = Settings()
```

### 5.2 Replace os.getenv() Calls

**Files to update:**
- `src/llm_model.py`
- `src/cost_tracking_wrapper.py`
- Any other files using `os.getenv()`

**Migration:**

```python
# Old
import os
api_key = os.getenv('OPENROUTER_API_KEY')

# New
from .settings import settings
api_key = settings.get_api_key('openrouter')
```

**Benefits:**
- Type-safe environment variables
- Automatic .env file loading
- SecretStr prevents accidental logging of keys
- Centralized configuration
- Validation of required environment variables

---

## Phase 6: Additional Models (Week 4)

**Priority:** LOW-MEDIUM - Nice to have improvements

### 6.1 Migrate ModelInfo TypedDict

**File:** `src/cost_tracking_wrapper.py`

```python
# Old
class ModelInfo(TypedDict, total=False):
    provider: str
    model_name: str
    input_cost_per_million: float

# New
class ModelInfo(BaseConfig):
    """Information about an LLM model."""
    provider: str
    model_name: str
    input_cost_per_million: Decimal
    output_cost_per_million: Decimal
    context_window: int | None = None
    supports_batch: bool = False
```

### 6.2 Pipeline Metadata Models

**File:** `src/pipeline.py`

```python
class PhaseMetadata(BaseConfig):
    """Metadata for a completed phase."""
    phase_name: str
    phase_type: str
    model_used: str
    started_at: str
    completed_at: str
    tokens_used: int
    cost: Decimal


class PipelineMetadata(BaseConfig):
    """Metadata for entire pipeline run."""
    book_name: str
    run_id: str
    started_at: str
    completed_at: str | None = None
    phases: list[PhaseMetadata] = Field(default_factory=list)
    total_cost: Decimal = Decimal('0.00')
    status: str  # 'running', 'completed', 'failed'
```

---

## Phase 7: Testing & Validation (Week 4-5)

### 7.1 Update Test Suite

**Files to update:**
- `tests/test_configuration.py`
- `tests/test_cost_tracking.py`
- `tests/test_llm_model.py`
- Any other relevant test files

**Testing strategy:**

1. **Parallel Testing:** Run old and new implementations side-by-side
2. **Validation Tests:** Ensure validation errors match expected behavior
3. **Serialization Tests:** Verify JSON encoding/decoding
4. **Integration Tests:** Test with real API responses (using fixtures)

**Example test:**

```python
import pytest
from pydantic import ValidationError
from src.config import PhaseConfig, PhaseType


def test_phase_config_validation():
    """Test PhaseConfig validates inputs correctly."""

    # Valid config
    config = PhaseConfig(
        phase_type=PhaseType.STANDARD,
        use_batch=True,
        batch_size=10,
        model_config_name="gpt-4"
    )
    assert config.batch_size == 10

    # Invalid batch_size
    with pytest.raises(ValidationError) as exc_info:
        PhaseConfig(
            phase_type=PhaseType.STANDARD,
            use_batch=True,
            batch_size=-5,  # Invalid
            model_config_name="gpt-4"
        )
    assert "batch_size must be > 0" in str(exc_info.value)

    # Missing required field
    with pytest.raises(ValidationError):
        PhaseConfig(phase_type=PhaseType.STANDARD)


def test_json_serialization():
    """Test models serialize/deserialize correctly."""
    config = PhaseConfig(
        phase_type=PhaseType.STANDARD,
        use_batch=False,
        model_config_name="gpt-4"
    )

    # Serialize
    json_data = config.model_dump_json()

    # Deserialize
    restored = PhaseConfig.model_validate_json(json_data)
    assert restored == config
```

### 7.2 Validation Checklist

**Per model migration:**
- [ ] All fields have correct types
- [ ] Required vs optional fields match original
- [ ] Validation logic matches original behavior
- [ ] Default values preserved
- [ ] JSON serialization works
- [ ] Existing tests pass
- [ ] Type checking passes (mypy)
- [ ] No performance regressions

**Overall:**
- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Type hints verified with mypy
- [ ] Linting passes (ruff)

---

## Phase 8: Cleanup & Documentation (Week 5)

### 8.1 Remove Old Code

**Actions:**
1. Remove old dataclass implementations
2. Remove manual validation code
3. Remove old `os.getenv()` calls
4. Clean up unused imports

### 8.2 Update Documentation

**Files to update:**
- `README.md` - Mention Pydantic usage
- `CONTRIBUTING.md` - Add Pydantic guidelines
- Code comments - Update where relevant
- Type stubs if any exist

**Example documentation:**

```markdown
## Configuration Models

All configuration in this project uses Pydantic v2 models for:
- Automatic validation
- Type safety
- JSON serialization
- Environment variable management

Example:
\`\`\`python
from src.config import PhaseConfig, PhaseType

config = PhaseConfig(
    phase_type=PhaseType.STANDARD,
    use_batch=True,
    batch_size=10,
    model_config_name="gpt-4"
)

# Automatic validation
config.batch_size = -1  # Raises ValidationError

# JSON serialization
json_str = config.model_dump_json()
restored = PhaseConfig.model_validate_json(json_str)
\`\`\`
```

### 8.3 Create Migration Guide

**New file:** `docs/PYDANTIC_USAGE.md`

Include:
- Overview of Pydantic usage in the project
- Common patterns and best practices
- How to create new models
- Validation patterns
- Testing guidelines
- Troubleshooting common issues

---

## Rollback Plan

### If Migration Needs to be Reverted

**Preparation:**
1. Create feature branch for migration: `feature/pydantic-migration`
2. Keep commits atomic and well-documented
3. Tag each phase completion: `migration-phase-1`, `migration-phase-2`, etc.

**Rollback Steps:**
1. Identify which phase caused issues
2. Revert commits to last stable phase tag
3. Document what went wrong
4. Plan remediation

**Risk Mitigation:**
- Migrate one phase at a time
- Run full test suite after each phase
- Keep old code until fully validated
- Use feature flags if needed for gradual rollout

---

## Success Metrics

### Quantitative
- [ ] Lines of code reduced: Target 200-300 lines
- [ ] Test coverage maintained: >90%
- [ ] Type checking coverage: 100% of migrated files
- [ ] Performance: No regressions (benchmark critical paths)

### Qualitative
- [ ] Developer experience improved (team feedback)
- [ ] Fewer validation-related bugs
- [ ] Easier to add new configuration options
- [ ] Better IDE support and autocomplete

---

## Timeline Summary

| Phase | Duration | Priority | Key Deliverable |
|-------|----------|----------|-----------------|
| 1. Setup | Week 1 | HIGH | Pydantic installed, base config ready |
| 2. Configuration | Week 1-2 | HIGH | Config models migrated, 150 lines saved |
| 3. Cost Tracking | Week 2 | HIGH | Cost models migrated |
| 4. API Models | Week 3 | MEDIUM-HIGH | Response models created |
| 5. Environment | Week 3 | MEDIUM | Settings model implemented |
| 6. Additional | Week 4 | LOW-MEDIUM | Misc models migrated |
| 7. Testing | Week 4-5 | HIGH | Full test coverage |
| 8. Cleanup | Week 5 | MEDIUM | Documentation complete |

**Total estimated time:** 5 weeks with regular progress

---

## Next Steps

1. **Review this plan** with the team
2. **Create GitHub issues** for each phase
3. **Set up feature branch:** `feature/pydantic-migration`
4. **Begin Phase 1** - Setup & Foundation
5. **Schedule weekly checkpoints** to review progress

---

## Resources

- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Migration Guide from v1 to v2](https://docs.pydantic.dev/latest/migration/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Validation Patterns](https://docs.pydantic.dev/latest/concepts/validators/)
- [JSON Schema Generation](https://docs.pydantic.dev/latest/concepts/json_schema/)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-21
**Owner:** Development Team
**Status:** Draft - Pending Approval

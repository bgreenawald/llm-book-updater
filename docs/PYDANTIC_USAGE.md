# Pydantic Usage Guide

This project uses Pydantic v2 models for configuration, API response validation, and cost tracking.

## Core Patterns

- Use `BaseConfig` from `src/pydantic_config.py` for all internal models.
- Prefer explicit types and optional fields with sensible defaults.
- Use `field_validator` for per-field checks and `model_validator` for cross-field rules.
- For API responses, use models in `src/api_models.py` to validate provider payloads.

## Configuration Models

- `src/config.py` defines `PhaseConfig`, `TwoStageModelConfig`, and `RunConfig`.
- Validation errors raise `pydantic.ValidationError` during instantiation.
- Defaults and derived fields are set in `model_validator` methods.

Example:

```python
from src.config import PhaseConfig, PhaseType

config = PhaseConfig(
    phase_type=PhaseType.MODERNIZE,
    use_batch=True,
    batch_size=10,
)
```

## Environment Settings

- `src/settings.py` provides a `Settings` model backed by `.env`.
- Use `settings.get_api_key("openrouter")` or other provider names.
- Use `settings.get_env("CUSTOM_ENV")` for non-standard keys.

## API Response Models

- `src/api_models.py` includes typed responses for OpenRouter, OpenAI, and Gemini.
- Use `Model.model_validate(payload)` to validate dict responses.

Example:

```python
from src.api_models import OpenRouterResponse

response = OpenRouterResponse.model_validate(payload)
content = response.choices[0].message["content"]
```

## Testing Guidance

- Use `pydantic.ValidationError` in tests that expect invalid input.
- Validate boundary conditions with simple, focused tests.

## Troubleshooting

- If a validation error appears unexpectedly, inspect `error.errors()` for field detail.
- For API changes, update the corresponding model in `src/api_models.py`.

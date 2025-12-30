"""Shared Pydantic configuration and base models."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration for all Pydantic models in the project."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=False,
        strict=False,
        validate_default=True,
    )

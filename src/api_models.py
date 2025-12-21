"""Pydantic models for LLM provider API responses."""

from pydantic import ConfigDict, Field

from src.pydantic_config import BaseConfig


class ApiBaseModel(BaseConfig):
    """Base model for API responses that may include extra fields."""

    model_config = BaseConfig.model_config | ConfigDict(extra="ignore")


class OpenRouterUsage(ApiBaseModel):
    """Token usage from OpenRouter API."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class OpenRouterChoice(ApiBaseModel):
    """Response choice from OpenRouter."""

    index: int
    message: dict[str, str]
    finish_reason: str | None = None


class OpenRouterResponse(ApiBaseModel):
    """Complete OpenRouter API response."""

    id: str | None = None
    model: str | None = None
    choices: list[OpenRouterChoice]
    usage: OpenRouterUsage | None = None
    created: int | None = None


class OpenAIUsage(ApiBaseModel):
    """Token usage from OpenAI API."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class OpenAIMessage(ApiBaseModel):
    """Message from OpenAI response."""

    role: str
    content: str | None = None


class OpenAIChoice(ApiBaseModel):
    """Response choice from OpenAI."""

    index: int
    message: OpenAIMessage
    finish_reason: str | None = None


class OpenAIResponse(ApiBaseModel):
    """Complete OpenAI API response."""

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str | None = None
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None


class GeminiUsageMetadata(ApiBaseModel):
    """Usage metadata from Gemini API."""

    prompt_token_count: int | None = Field(None, alias="promptTokenCount")
    candidates_token_count: int | None = Field(None, alias="candidatesTokenCount")
    total_token_count: int | None = Field(None, alias="totalTokenCount")


class GeminiContent(ApiBaseModel):
    """Content from Gemini response."""

    parts: list[dict] = Field(default_factory=list)
    role: str | None = None


class GeminiCandidate(ApiBaseModel):
    """Candidate response from Gemini."""

    content: GeminiContent | None = None
    finish_reason: str | None = Field(None, alias="finishReason")
    index: int | None = None


class GeminiResponse(ApiBaseModel):
    """Complete Gemini API response."""

    candidates: list[GeminiCandidate] = Field(default_factory=list)
    usage_metadata: GeminiUsageMetadata | None = Field(None, alias="usageMetadata")
    model_version: str | None = Field(None, alias="modelVersion")


class BatchRequest(ApiBaseModel):
    """Generic batch request format."""

    custom_id: str
    method: str = "POST"
    url: str
    body: dict


class BatchResponse(ApiBaseModel):
    """Generic batch response format."""

    id: str | None = None
    custom_id: str
    response: dict | None = None
    error: dict | None = None

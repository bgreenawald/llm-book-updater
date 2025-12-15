import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.provider import Provider
from src.constants import (
    DEFAULT_OPENROUTER_BACKOFF_FACTOR,
    DEFAULT_OPENROUTER_MAX_RETRIES,
    DEFAULT_OPENROUTER_RETRY_DELAY,
    GEMINI_DEFAULT_TEMPERATURE,
    LLM_DEFAULT_TEMPERATURE,
    OPENAI_BATCH_DEFAULT_TIMEOUT,
    OPENAI_BATCH_POLLING_INTERVAL,
    OPENROUTER_POOL_CONNECTIONS,
    OPENROUTER_POOL_MAXSIZE,
    OPENROUTER_REQUEST_TIMEOUT,
    PROMPT_PREVIEW_MAX_LENGTH,
)
from src.cost_tracking_wrapper import register_generation_model_info

# Load environment variables from .env to ensure API keys are available
load_dotenv(override=True)

# Initialize module-level logger
module_logger = logger


@dataclass
class ModelConfig:
    """Configuration for a model, including provider and model identifier."""

    provider: Provider
    model_id: str
    # Provider-specific model name (for direct SDK calls)
    provider_model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Set provider_model_name if not specified."""
        if self.provider_model_name is None:
            if self.provider == Provider.OPENAI:
                # Extract OpenAI model name from OpenRouter format
                if "/" in self.model_id:
                    self.provider_model_name = self.model_id.split("/", 1)[1]
                else:
                    self.provider_model_name = self.model_id
            elif self.provider == Provider.GEMINI:
                # Extract Gemini model name from OpenRouter format
                if "/" in self.model_id:
                    self.provider_model_name = self.model_id.split("/", 1)[1]
                else:
                    self.provider_model_name = self.model_id
            else:
                # For OpenRouter, use the full model_id
                self.provider_model_name = self.model_id


# Model constants with provider information
GROK_3_MINI = ModelConfig(Provider.OPENROUTER, "x-ai/grok-3-mini")
GEMINI_FLASH = ModelConfig(Provider.GEMINI, "google/gemini-2.5-flash", "gemini-2.5-flash")
GEMINI_PRO = ModelConfig(Provider.GEMINI, "google/gemini-2.5-pro", "gemini-2.5-pro")
DEEPSEEK = ModelConfig(Provider.OPENROUTER, "deepseek/deepseek-r1-0528")
OPENAI_04_MINI = ModelConfig(Provider.OPENAI, "openai/o4-mini-high", "o4-mini")
CLAUDE_4_SONNET = ModelConfig(Provider.OPENROUTER, "anthropic/claude-sonnet-4")
GEMINI_FLASH_LITE = ModelConfig(Provider.GEMINI, "google/gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite")
KIMI_K2 = ModelConfig(Provider.OPENROUTER, "moonshotai/kimi-k2:free")
# Claude API models (direct Anthropic API)
CLAUDE_OPUS_4_5 = ModelConfig(Provider.CLAUDE, "claude-opus-4-5")
CLAUDE_SONNET_4_5 = ModelConfig(Provider.CLAUDE, "claude-sonnet-4-5")
CLAUDE_HAIKU_4_5 = ModelConfig(Provider.CLAUDE, "claude-haiku-4-5")


class LlmModelError(Exception):
    """Custom exception for LLM model errors."""

    pass


class ProviderClient(ABC):
    """Abstract base class for provider-specific LLM clients."""

    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """
        Check if this provider supports batch processing.

        Returns:
            bool: True if batch processing is supported, False otherwise
        """
        pass

    @abstractmethod
    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call.

        Returns:
            Tuple of (response_content, generation_id)
        """
        pass

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Make batch chat completion calls.

        Args:
            requests: List of request dictionaries, each containing:
                - system_prompt: System prompt for the request
                - user_prompt: User prompt for the request
                - metadata: Optional metadata to include in response
            model_name: Name of the model to use
            temperature: Temperature setting for the model
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries, each containing:
                - content: Response content
                - generation_id: Generation ID for tracking
                - metadata: Original metadata from request
        """
        # Default implementation: process sequentially
        responses = []
        for request in requests:
            content, generation_id = self.chat_completion(
                system_prompt=request["system_prompt"],
                user_prompt=request["user_prompt"],
                model_name=model_name,
                temperature=temperature,
                **kwargs,
            )
            responses.append(
                {"content": content, "generation_id": generation_id, "metadata": request.get("metadata", {})}
            )
        return responses


class OpenRouterClient(ProviderClient):
    """Client for OpenRouter API calls with connection pooling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = DEFAULT_OPENROUTER_MAX_RETRIES,
        retry_delay: float = DEFAULT_OPENROUTER_RETRY_DELAY,
        backoff_factor: float = DEFAULT_OPENROUTER_BACKOFF_FACTOR,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        # Initialize session with connection pooling
        self._session = requests.Session()

        # Configure retry strategy for the session
        # Note: This handles automatic retries at the connection level
        # We still keep our application-level retry logic for custom backoff
        retry_strategy = Retry(
            total=0,  # Disable automatic retries; we handle retries manually
            connect=3,  # But allow connection retries for network issues
            read=3,  # And read retries for incomplete responses
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        )

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=OPENROUTER_POOL_CONNECTIONS,
            pool_maxsize=OPENROUTER_POOL_MAXSIZE,
            max_retries=retry_strategy,
        )

        # Mount adapter for both http and https
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Set default headers for all requests
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def close(self) -> None:
        """Close the session and release connection pool resources."""
        if hasattr(self, "_session"):
            self._session.close()

    @property
    def supports_batch(self) -> bool:
        """OpenRouter does not support batch processing."""
        return False

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, requests.exceptions.RequestException):
            if hasattr(error, "response") and error.response is not None:
                status_code = error.response.status_code
                return status_code >= 500 or status_code == 429
            return True
        return False

    def _make_api_call(self, data: dict) -> dict:
        """Makes a single API call to OpenRouter with retry logic using session.

        Args:
            data: Request payload to send to the API

        Returns:
            Response JSON from the API

        Note:
            Headers are already set in the session, so we don't need to pass them.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url=f"{self.base_url}/chat/completions",
                    data=json.dumps(obj=data),
                    timeout=OPENROUTER_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries and self._should_retry(error=e):
                    delay = self.retry_delay * (self.backoff_factor**attempt)
                    module_logger.warning(
                        f"API call failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    break
            except json.JSONDecodeError as e:
                last_error = e
                module_logger.error(f"Failed to parse response JSON: {e}")
                break
            except Exception as e:
                last_error = e
                module_logger.error(f"Unexpected error during API call: {e}")
                break

        error_msg = f"API call failed after {self.max_retries + 1} attempts"
        if last_error:
            error_msg += f": {last_error}"

        module_logger.error(error_msg)
        raise LlmModelError(error_msg) from last_error

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenRouter API."""
        # GPT-5 models (and variants) currently do not accept custom temperature
        # values. Detect and omit temperature to avoid 400 errors from upstream.
        provider_model = model_name.split("/", 1)[1] if "/" in model_name else model_name
        is_gpt5_series_model = provider_model.lower().startswith("gpt-5")

        data: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        }
        if not is_gpt5_series_model:
            data["temperature"] = temperature

        resp_data = self._make_api_call(data=data)

        choices = resp_data.get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError(f"Empty or malformed response: {resp_data}")

        content = choices[0]["message"]["content"]
        finish_reason = choices[0].get("finish_reason", "unknown")

        if finish_reason == "length":
            module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")

        generation_id = resp_data.get("id", "unknown")
        return content, generation_id


class OpenAIClient(ProviderClient):
    """Client for OpenAI SDK calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def supports_batch(self) -> bool:
        """OpenAI supports batch processing."""
        return True

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise LlmModelError(f"OpenAI SDK not available: {e}") from e
        return self._client

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenAI SDK."""
        client = self._get_client()

        try:
            # GPT-5 models (and variants) currently do not support custom
            # temperature values; only the default is allowed. Omit the
            # temperature parameter for these models to avoid 400 errors.
            is_gpt5_series_model = model_name.lower().startswith("gpt-5")

            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "instructions": system_prompt,
                "input": user_prompt,
            }

            if not is_gpt5_series_model:
                request_kwargs["temperature"] = temperature

            response = client.responses.create(**request_kwargs, **kwargs)

            # Extract content from the new responses API format
            # Response format: response.output[0] should be a message with content[0].text
            if not response.output:
                raise ValueError("Empty response output from OpenAI")

            # Find the first message object in output (skip reasoning items)
            output_message = None
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "message":
                    output_message = output_item
                    break

            if output_message is None:
                raise ValueError("No message output found in OpenAI response")

            if not output_message.content or not output_message.content[0].text:
                raise ValueError("Empty response content from OpenAI")

            content = output_message.content[0].text

            # Check if response was truncated (status might indicate this)
            if (
                getattr(output_message, "status", None) == "incomplete"
                or getattr(response, "status", None) == "incomplete"
            ):
                module_logger.warning("Response truncated: consider increasing max_tokens or reviewing model limits")

            # Register token usage for cost tracking if available
            try:
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
                completion_tokens = getattr(usage, "output_tokens", None) if usage is not None else None

                if prompt_tokens is not None or completion_tokens is not None:
                    register_generation_model_info(
                        generation_id=response.id,
                        model=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            except (AttributeError, TypeError, ValueError) as e:
                # Usage extraction is optional; log but don't fail the call
                module_logger.debug(f"Failed to extract usage metadata for {model_name}: {e}")

            return content, response.id

        except Exception as e:
            module_logger.error(f"OpenAI API call failed: {e}")
            raise LlmModelError(f"OpenAI API call failed: {e}") from e

    def _create_batch_jsonl(self, requests: list[dict[str, Any]], model_name: str, temperature: float, **kwargs) -> str:
        """
        Create JSONL content for batch processing using the OpenAI Responses API.

        Args:
            requests: List of request dictionaries containing system_prompt, user_prompt, metadata
            model_name: Name of the model to use
            temperature: Temperature setting for the model
            **kwargs: Additional parameters (e.g., effort under reasoning)

        Returns:
            str: JSONL formatted string for batch processing
        """
        import json

        jsonl_lines = []
        for i, request in enumerate(requests):
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("user_prompt", "")
            metadata = request.get("metadata", {})

            # Handle GPT-5 models that don't support custom temperature
            is_gpt5_series_model = model_name.lower().startswith("gpt-5")

            # Build request body for Responses API
            batch_body: Dict[str, Any] = {
                "model": model_name,
                "input": user_prompt,
            }

            if system_prompt:
                batch_body["instructions"] = system_prompt

            # Only add temperature for non-GPT-5 models
            if not is_gpt5_series_model:
                batch_body["temperature"] = temperature

            # Optional: pass through reasoning effort if provided (Responses API)
            effort = kwargs.get("effort")
            if effort is not None:
                batch_body["reasoning"] = {"effort": effort}

            batch_request = {
                "custom_id": f"request_{i}_{metadata.get('id', '')}",
                "method": "POST",
                "url": "/v1/responses",
                "body": batch_body,
            }

            jsonl_lines.append(json.dumps(batch_request))

        return "\n".join(jsonl_lines)

    def _poll_batch_job(self, client, job_id: str, timeout_seconds: int = 3600 * 24) -> bool:
        """
        Poll batch job until completion or timeout.

        Args:
            client: OpenAI client
            job_id: ID of the batch job
            timeout_seconds: Maximum time to wait for completion (default 1 hour)

        Returns:
            bool: True if job completed successfully, False otherwise
        """
        import time

        start_time = time.time()
        poll_interval = OPENAI_BATCH_POLLING_INTERVAL

        while time.time() - start_time < timeout_seconds:
            try:
                batch_job = client.batches.retrieve(job_id)
                status = batch_job.status

                module_logger.info(f"Batch job {job_id} status: {status}")

                if status == "completed":
                    # Additional check: ensure the job actually has results
                    if hasattr(batch_job, "request_counts"):
                        total = getattr(batch_job.request_counts, "total", 0)
                        completed = getattr(batch_job.request_counts, "completed", 0)
                        failed = getattr(batch_job.request_counts, "failed", 0)
                        module_logger.info(
                            f"Batch job request counts - Total: {total}, Completed: {completed}, Failed: {failed}"
                        )

                        # If all requests failed, treat as failed job
                        if total > 0 and failed == total:
                            module_logger.error(f"All {total} requests in batch job failed")
                            return False
                    return True
                elif status in ["failed", "expired", "cancelled", "cancelling"]:
                    error_msg = f"Batch job failed with status: {status}"
                    if hasattr(batch_job, "errors") and batch_job.errors:
                        error_msg += f" - Errors: {batch_job.errors}"
                    module_logger.error(error_msg)
                    return False
                elif status in ["validating", "in_progress", "finalizing"]:
                    # These are expected intermediate states, continue polling
                    module_logger.debug(f"Batch job in progress with status: {status}")
                else:
                    # Unexpected status
                    module_logger.warning(f"Unexpected batch job status: {status}")

                time.sleep(poll_interval)

            except (AttributeError, TypeError) as e:
                # Polling errors indicate API response structure issues
                module_logger.error(f"Error polling batch job: {e}")
                return False

        module_logger.error(f"Batch job timed out after {timeout_seconds} seconds")
        return False

    def _parse_batch_results(
        self, result_content: str, original_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Parse batch job results and match them with original requests.

        Args:
            result_content: JSONL content from batch job results
            original_requests: Original request list for metadata matching

        Returns:
            List of response dictionaries
        """
        import json
        import time

        responses = []
        result_lines = result_content.strip().split("\n")

        # Create mapping from custom IDs to original metadata
        id_to_metadata = {}
        for i, request in enumerate(original_requests):
            metadata = request.get("metadata", {})
            custom_id = f"request_{i}_{metadata.get('id', '')}"
            id_to_metadata[custom_id] = metadata

        for line in result_lines:
            if not line.strip():
                continue

            try:
                result = json.loads(line)

                # Extract response data
                custom_id = result.get("custom_id", "")

                # Check for errors in the result - errors can be in multiple places
                error_info = None

                # Check for top-level error
                if "error" in result and result["error"]:
                    error_info = result["error"]
                # Check for response-level error (HTTP status != 200)
                elif "response" in result:
                    response = result["response"]
                    if response.get("status_code", 200) != 200:
                        # Error is in response body
                        response_body = response.get("body", {})
                        if "error" in response_body:
                            error_info = response_body["error"]
                        else:
                            error_info = {"message": f"HTTP {response.get('status_code')} error", "type": "http_error"}

                if error_info:
                    error_message = (
                        error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
                    )
                    module_logger.error(f"Error in batch result {custom_id}: {error_message}")
                    responses.append(
                        {
                            "content": f"Error: {error_message}",
                            "generation_id": f"openai_batch_error_{custom_id}",
                            "metadata": id_to_metadata.get(custom_id, {}),
                        }
                    )
                    continue

                # Extract successful response (Responses API format)
                response_body = result.get("response", {}).get("body", {})

                # Debug logging to see what we're getting
                module_logger.debug(
                    f"Batch result structure for {custom_id}: response_body keys: {list(response_body.keys())}"
                )

                content = ""
                output_items = response_body.get("output", []) or []
                # Find the first assistant message and extract its text part
                for item in output_items:
                    if isinstance(item, dict) and item.get("type") == "message":
                        content_parts = item.get("content", []) or []
                        for part in content_parts:
                            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                                content = part.get("text", "") or ""
                                break
                        if content:
                            break

                if not content:
                    module_logger.warning(
                        f"Empty content in batch response for {custom_id}, output keys: {
                            [i.get('type') for i in output_items if isinstance(i, dict)]
                        }"
                    )

                generation_id = response_body.get("id", f"openai_batch_{custom_id}")

                # Get original metadata
                metadata = id_to_metadata.get(custom_id, {})

                # Track token usage if available
                try:
                    usage = response_body.get("usage", {})
                    if usage:
                        # Responses API uses input_tokens/output_tokens
                        prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
                        completion_tokens = usage.get("output_tokens", usage.get("completion_tokens"))

                        if prompt_tokens is not None or completion_tokens is not None:
                            register_generation_model_info(
                                generation_id=generation_id,
                                model=response_body.get("model", "unknown"),
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                is_batch=True,  # This is batch processing, apply 50% discount
                            )
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    # Usage tracking is optional for batch; log but don't fail
                    module_logger.debug(f"Failed to track batch usage for {generation_id}: {e}")

                responses.append({"content": content, "generation_id": generation_id, "metadata": metadata})

            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                module_logger.error(f"Error parsing batch result line: {e}")
                responses.append(
                    {
                        "content": f"Error parsing result: {str(e)}",
                        "generation_id": f"openai_batch_parse_error_{int(time.time())}",
                        "metadata": {},
                    }
                )

        return responses

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        batch_timeout: int = OPENAI_BATCH_DEFAULT_TIMEOUT,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Process batch chat completion requests using OpenAI's true batch API.

        This implementation creates a batch job with all requests in JSONL format,
        submits it to the OpenAI API, polls for completion, and retrieves results.
        Batch processing provides 50% cost savings compared to synchronous API calls.

        Args:
            requests: List of request dictionaries
            model_name: Name of the model to use
            temperature: Temperature setting for the model
            batch_timeout: Maximum time to wait for batch job completion (seconds)
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries
        """
        client = self._get_client()

        try:
            import os
            import tempfile

            module_logger.info(f"Starting OpenAI batch processing for {len(requests)} requests")

            # Create JSONL content for batch processing
            jsonl_content = self._create_batch_jsonl(requests, model_name, temperature, **kwargs)

            # Log sample of batch content for debugging
            jsonl_lines = jsonl_content.split("\n")
            if jsonl_lines:
                module_logger.debug(f"Sample batch request (first item): {jsonl_lines[0][:500]}")
                module_logger.info(f"Batch contains {len(jsonl_lines)} requests")

            # Create temporary file for batch requests
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
                temp_file.write(jsonl_content)
                temp_file_path = temp_file.name

            try:
                # Upload the batch requests file
                module_logger.info("Uploading batch requests file...")
                with open(temp_file_path, "rb") as file:
                    batch_file = client.files.create(file=file, purpose="batch")

                # Create the batch job
                module_logger.info(f"Creating batch job with model {model_name}...")
                batch_job = client.batches.create(
                    input_file_id=batch_file.id, endpoint="/v1/responses", completion_window="24h"
                )

                module_logger.info(f"Created batch job: {batch_job.id}")

                # Poll for job completion
                module_logger.info(f"Polling batch job for completion (timeout: {batch_timeout}s)...")
                poll_result = self._poll_batch_job(client, batch_job.id, batch_timeout)

                # Always retrieve the job to check for errors, even if polling "failed"
                completed_job = client.batches.retrieve(batch_job.id)

                if not poll_result:
                    module_logger.error("Batch job did not complete successfully or all requests failed")

                    # Try to get error details before falling back
                    if completed_job.error_file_id:
                        try:
                            error_file = client.files.content(completed_job.error_file_id)
                            error_content = error_file.read().decode("utf-8")
                            module_logger.error(f"Batch job error details:\n{error_content}")
                        except Exception as e:
                            module_logger.error(f"Failed to retrieve error file: {e}")

                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

                # Log detailed job information for debugging
                module_logger.info(
                    f"Batch job details - Status: {completed_job.status}, "
                    f"Request counts: {completed_job.request_counts}, "
                    f"Has output_file_id: {bool(completed_job.output_file_id)}, "
                    f"Has error_file_id: {bool(completed_job.error_file_id)}"
                )

                # Check for error file first
                if completed_job.error_file_id:
                    module_logger.warning(f"Batch job has error file: {completed_job.error_file_id}")
                    try:
                        error_file = client.files.content(completed_job.error_file_id)
                        error_content = error_file.read().decode("utf-8")
                        module_logger.error(f"Batch job errors:\n{error_content[:1000]}")  # Log first 1000 chars
                    except Exception as e:
                        module_logger.error(f"Failed to retrieve error file: {e}")

                # Retrieve results
                if completed_job.output_file_id:
                    module_logger.info(f"Downloading batch results from {completed_job.output_file_id}...")

                    result_file = client.files.content(completed_job.output_file_id)
                    result_content = result_file.read().decode("utf-8")

                    # Debug logging for batch content
                    module_logger.debug(f"Raw batch result content sample: {result_content[:500]}...")

                    # Parse and return results
                    responses = self._parse_batch_results(result_content, requests)

                    module_logger.info(f"Successfully processed {len(responses)} batch responses")
                    return responses
                else:
                    # Log more details about why there's no output file
                    failed_count = (
                        completed_job.request_counts.failed
                        if hasattr(completed_job.request_counts, "failed")
                        else "unknown"
                    )
                    module_logger.error(
                        f"No output file found in completed batch job. "
                        f"Job status: {completed_job.status}, "
                        f"Failed requests: {failed_count}"
                    )
                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    # Cleanup errors are non-critical but should be visible
                    module_logger.debug(f"Failed to clean up temp file {temp_file_path}: {e}")

        except ImportError as e:
            module_logger.error(f"OpenAI SDK not available for batch processing: {e}")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)
        except Exception as e:
            module_logger.error(f"OpenAI batch API call failed: {e}")
            module_logger.exception("Batch processing error details:")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)


class GeminiClient(ProviderClient):
    """Client for Google Gemini SDK calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def supports_batch(self) -> bool:
        """Gemini supports batch processing."""
        return True

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                raise LlmModelError(f"Google GenAI SDK not available: {e}") from e
        return self._client

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using Gemini SDK."""
        client = self._get_client()

        try:
            from google.genai import types

            # Configure generation settings
            # Remove unsupported parameters (e.g., reasoning)
            kwargs.pop("reasoning", None)

            # Extract contents if provided in kwargs (for advanced content structures)
            contents = kwargs.pop("contents", user_prompt)

            # Build config_kwargs from remaining kwargs, avoiding duplicates
            config_kwargs = dict(kwargs)

            # Explicitly set system_instruction and temperature, overriding any in kwargs
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            else:
                # Remove system_instruction if it exists in kwargs and no system_prompt provided
                config_kwargs.pop("system_instruction", None)

            # Always set temperature explicitly to avoid duplicates
            config_kwargs["temperature"] = temperature

            # Create the config with deduplicated parameters
            config = types.GenerateContentConfig(**config_kwargs)

            response = client.models.generate_content(model=model_name, contents=contents, config=config)

            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Gemini doesn't provide a completion ID in the same way, so we'll generate one
            generation_id = f"gemini_{int(time.time())}"

            # Extract and register token usage for cost tracking if available
            try:
                usage_metadata = getattr(response, "usage_metadata", None)
                if usage_metadata:
                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
                    completion_tokens = getattr(usage_metadata, "candidates_token_count", None)

                    if prompt_tokens is not None or completion_tokens is not None:
                        register_generation_model_info(
                            generation_id=generation_id,
                            model=model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
            except Exception as e:
                # Log but don't fail if we can't extract usage metadata
                module_logger.debug(f"Could not extract Gemini usage metadata: {e}")

            return response.text, generation_id

        except Exception as e:
            module_logger.error(f"Gemini API call failed: {e}")
            raise LlmModelError(f"Gemini API call failed: {e}") from e

    def _create_batch_jsonl(self, requests: list[dict[str, Any]]) -> str:
        """
        Create JSONL content for batch processing following Gemini API specification.

        Args:
            requests: List of request dictionaries containing system_prompt, user_prompt, metadata

        Returns:
            str: JSONL formatted string for batch processing
        """
        import json

        jsonl_lines = []
        for i, request in enumerate(requests):
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("user_prompt", "")
            metadata = request.get("metadata", {})

            # Create the user content parts
            user_parts = []
            if user_prompt:
                user_parts.append({"text": user_prompt})

            # Create the batch request structure according to GenerateContentRequest spec
            batch_request = {
                "key": f"request_{i}",
                "request": {"contents": [{"parts": user_parts}]},
                "metadata": metadata,  # Store metadata for result matching
            }

            # Add system instruction separately if present
            if system_prompt:
                batch_request["request"]["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            jsonl_lines.append(json.dumps(batch_request))

        return "\n".join(jsonl_lines)

    def _poll_batch_job(self, client, job_name: str, timeout_seconds: int = 3600 * 24) -> bool:
        """
        Poll batch job until completion or timeout.

        Args:
            client: GenAI client
            job_name: Name of the batch job
            timeout_seconds: Maximum time to wait for completion (default 1 hour)

        Returns:
            bool: True if job completed successfully, False otherwise
        """
        import time

        start_time = time.time()
        poll_interval = OPENAI_BATCH_POLLING_INTERVAL

        while time.time() - start_time < timeout_seconds:
            try:
                job = client.batches.get(name=job_name)
                state = job.state.name if hasattr(job.state, "name") else str(job.state)

                module_logger.info(f"Batch job {job_name} state: {state}")

                if state == "JOB_STATE_SUCCEEDED":
                    return True
                elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                    module_logger.error(f"Batch job failed with state: {state}")
                    return False

                time.sleep(poll_interval)

            except (AttributeError, TypeError) as e:
                # Polling errors indicate API response structure issues
                module_logger.error(f"Error polling batch job: {e}")
                return False

        module_logger.error(f"Batch job timed out after {timeout_seconds} seconds")
        return False

    def _parse_batch_results(
        self, result_content: str, original_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Parse batch job results and match them with original requests.

        Args:
            result_content: JSONL content from batch job results
            original_requests: Original request list for metadata matching

        Returns:
            List of response dictionaries
        """
        import json

        responses = []
        result_lines = result_content.strip().split("\n")

        # Create mapping from request keys to original metadata
        key_to_metadata = {}
        for i, request in enumerate(original_requests):
            key_to_metadata[f"request_{i}"] = request.get("metadata", {})

        for line in result_lines:
            try:
                result = json.loads(line)

                # Extract response data
                key = result.get("key", "")
                response_data = result.get("response", {})

                # Get the generated content
                candidates = response_data.get("candidates", [])
                content = ""
                if candidates:
                    candidate = candidates[0]
                    content_parts = candidate.get("content", {}).get("parts", [])
                    if content_parts:
                        content = content_parts[0].get("text", "")

                # Generate a unique ID
                generation_id = f"gemini_batch_{int(time.time())}_{key}"

                # Get original metadata
                metadata = key_to_metadata.get(key, {})

                # Track token usage if available
                try:
                    usage_metadata = response_data.get("usageMetadata", {})
                    if usage_metadata:
                        prompt_tokens = usage_metadata.get("promptTokenCount")
                        completion_tokens = usage_metadata.get("candidatesTokenCount")

                        if prompt_tokens is not None or completion_tokens is not None:
                            register_generation_model_info(
                                generation_id=generation_id,
                                model="gemini",  # Model name from batch job
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                is_batch=True,  # This is batch processing, apply 50% discount
                            )
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    # Usage tracking is optional for batch; log but don't fail
                    module_logger.debug(f"Failed to track Gemini batch usage for {generation_id}: {e}")

                responses.append({"content": content, "generation_id": generation_id, "metadata": metadata})

            except (json.JSONDecodeError, KeyError, ValueError, IndexError, AttributeError) as e:
                module_logger.error(f"Error parsing batch result line: {e}")
                # Add error response
                responses.append(
                    {
                        "content": f"Error parsing result: {str(e)}",
                        "generation_id": f"gemini_batch_parse_error_{int(time.time())}",
                        "metadata": {},
                    }
                )

        return responses

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        batch_timeout: int = 3600 * 24,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Process batch chat completion requests using Gemini's true batch job API.

        This implementation creates a batch job with all requests in JSONL format,
        submits it to the Gemini API, polls for completion, and retrieves results.
        Batch processing provides 50% cost savings compared to synchronous API calls.

        Args:
            requests: List of request dictionaries
            model_name: Name of the model to use
            temperature: Temperature setting for the model
            batch_timeout: Maximum time to wait for batch job completion (seconds)
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries
        """
        client = self._get_client()

        try:
            import os
            import tempfile

            from google.genai.types import CreateBatchJobConfig

            # Remove unsupported parameters
            kwargs.pop("reasoning", None)

            module_logger.info(f"Starting Gemini batch processing for {len(requests)} requests")

            # Create JSONL content for batch processing
            jsonl_content = self._create_batch_jsonl(requests)

            # Create temporary file for batch requests
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
                temp_file.write(jsonl_content)
                temp_file_path = temp_file.name

            try:
                # Upload the batch requests file
                module_logger.info("Uploading batch requests file...")
                uploaded_file = client.files.upload(file=temp_file_path, config={"mimeType": "application/jsonl"})

                # Create batch configuration
                batch_config = CreateBatchJobConfig(
                    display_name=f"llm_book_updater_batch_{int(time.time())}",
                )

                # Add temperature to the config if supported
                if temperature != GEMINI_DEFAULT_TEMPERATURE:  # Only set if different from default
                    # Note: Temperature might need to be set differently in batch config
                    # The exact parameter name may vary - check latest API docs
                    pass

                # Create the batch job
                module_logger.info(f"Creating batch job with model {model_name}...")
                batch_job = client.batches.create(
                    model=model_name,
                    src=uploaded_file.name,
                    config=batch_config,
                )

                module_logger.info(f"Created batch job: {batch_job.name}")

                # Poll for job completion
                module_logger.info(f"Polling batch job for completion (timeout: {batch_timeout}s)...")
                if not self._poll_batch_job(client, batch_job.name, batch_timeout):
                    module_logger.error("Batch job did not complete successfully")
                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

                # Get the completed job to retrieve results
                completed_job = client.batches.get(name=batch_job.name)

                # Download results
                if hasattr(completed_job, "dest") and hasattr(completed_job.dest, "file_name"):
                    result_file_name = completed_job.dest.file_name
                    module_logger.info(f"Downloading batch results from {result_file_name}...")

                    result_bytes = client.files.download(file=result_file_name)
                    result_content = result_bytes.decode("utf-8")

                    # Parse and return results
                    responses = self._parse_batch_results(result_content, requests)

                    module_logger.info(f"Successfully processed {len(responses)} batch responses")
                    return responses
                else:
                    module_logger.error("No result file found in completed batch job")
                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    # Cleanup errors are non-critical but should be visible
                    module_logger.debug(f"Failed to clean up temp file {temp_file_path}: {e}")

        except ImportError as e:
            module_logger.error(f"Google GenAI SDK not available for batch processing: {e}")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)
        except Exception as e:
            module_logger.error(f"Gemini batch API call failed: {e}")
            module_logger.exception("Batch processing error details:")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)


class ClaudeClient(ProviderClient):
    """Client for Claude (Anthropic) SDK calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def supports_batch(self) -> bool:
        """Claude supports batch processing."""
        return True

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise LlmModelError(f"Anthropic SDK not available: {e}") from e
        return self._client

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using Anthropic SDK."""
        client = self._get_client()

        try:
            # Build the request parameters
            request_params: dict[str, Any] = {
                "model": model_name,
                "max_tokens": kwargs.pop("max_tokens", 65536),  # 64K for Claude 4.5 book chapters
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": temperature,
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            # Pass through any additional kwargs
            request_params.update(kwargs)

            response = client.messages.create(**request_params)

            # Extract content from response
            if not response.content or not response.content[0].text:
                raise ValueError("Empty response from Claude")

            content = response.content[0].text

            # Check stop reason for warnings
            if response.stop_reason == "max_tokens":
                module_logger.warning("Response truncated: consider increasing max_tokens")

            # Register token usage for cost tracking if available
            try:
                usage = response.usage
                if usage:
                    register_generation_model_info(
                        generation_id=response.id,
                        model=model_name,
                        prompt_tokens=usage.input_tokens,
                        completion_tokens=usage.output_tokens,
                    )
            except (AttributeError, TypeError, ValueError) as e:
                # Usage extraction is optional; log but don't fail the call
                module_logger.debug(f"Failed to extract usage metadata for {model_name}: {e}")

            return content, response.id

        except Exception as e:
            module_logger.error(f"Claude API call failed: {e}")
            raise LlmModelError(f"Claude API call failed: {e}") from e

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        temperature: float,
        batch_timeout: int = 3600 * 24,  # 24 hours default
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Process batch chat completion requests using Claude's Message Batches API.

        This implementation creates a batch job with all requests, submits it to the
        Anthropic API, polls for completion, and retrieves results. Batch processing
        provides 50% cost savings compared to synchronous API calls.

        Args:
            requests: List of request dictionaries
            model_name: Name of the model to use
            temperature: Temperature setting for the model
            batch_timeout: Maximum time to wait for batch job completion (seconds)
            **kwargs: Additional arguments (e.g., max_tokens)

        Returns:
            List of response dictionaries
        """
        client = self._get_client()

        try:
            import time

            module_logger.info(f"Starting Claude batch processing for {len(requests)} requests")

            # Build batch requests
            batch_requests = []
            for i, request in enumerate(requests):
                system_prompt = request.get("system_prompt", "")
                user_prompt = request.get("user_prompt", "")
                metadata = request.get("metadata", {})

                # Build the params for this request
                params: dict[str, Any] = {
                    "model": model_name,
                    "max_tokens": kwargs.get("max_tokens", 65536),  # 64K for Claude 4.5
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": temperature,
                }

                # Add system prompt if provided
                if system_prompt:
                    params["system"] = system_prompt

                # Create batch request with custom_id
                batch_request = {
                    "custom_id": f"request_{i}_{metadata.get('id', '')}",
                    "params": params,
                }

                batch_requests.append(batch_request)

            # Create the batch
            module_logger.info(f"Creating Claude batch job with model {model_name}...")
            message_batch = client.messages.batches.create(requests=batch_requests)

            module_logger.info(f"Created batch job: {message_batch.id}")

            # Poll for batch completion
            module_logger.info(f"Polling batch job for completion (timeout: {batch_timeout}s)...")
            start_time = time.time()
            poll_interval = 60  # Poll every 60 seconds

            while time.time() - start_time < batch_timeout:
                message_batch = client.messages.batches.retrieve(message_batch.id)

                module_logger.info(f"Batch job {message_batch.id} status: {message_batch.processing_status}")

                if message_batch.processing_status == "ended":
                    break
                elif message_batch.processing_status in ["canceling", "canceled"]:
                    module_logger.error("Batch job was canceled")
                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

                time.sleep(poll_interval)
            else:
                # Timeout reached
                module_logger.error(f"Batch job timed out after {batch_timeout} seconds")
                # Fall back to sequential processing
                return super().batch_chat_completion(requests, model_name, temperature, **kwargs)

            # Retrieve and process results
            module_logger.info("Downloading batch results...")

            # Create mapping from custom IDs to original metadata
            id_to_metadata = {}
            for i, request in enumerate(requests):
                metadata = request.get("metadata", {})
                custom_id = f"request_{i}_{metadata.get('id', '')}"
                id_to_metadata[custom_id] = metadata

            # Stream results
            responses = []
            for result in client.messages.batches.results(message_batch.id):
                custom_id = result.custom_id
                metadata = id_to_metadata.get(custom_id, {})

                if result.result.type == "succeeded":
                    # Extract successful response
                    message = result.result.message
                    content = message.content[0].text if message.content else ""
                    generation_id = message.id

                    # Track token usage if available
                    try:
                        usage = message.usage
                        if usage:
                            register_generation_model_info(
                                generation_id=generation_id,
                                model=model_name,
                                prompt_tokens=usage.input_tokens,
                                completion_tokens=usage.output_tokens,
                                is_batch=True,  # 50% discount for batch
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        module_logger.debug(f"Failed to track batch usage for {generation_id}: {e}")

                    responses.append({"content": content, "generation_id": generation_id, "metadata": metadata})

                elif result.result.type == "errored":
                    # Handle error result
                    error = result.result.error
                    error_message = error.message if hasattr(error, "message") else str(error)
                    module_logger.error(f"Error in batch result {custom_id}: {error_message}")
                    responses.append(
                        {
                            "content": f"Error: {error_message}",
                            "generation_id": f"claude_batch_error_{custom_id}",
                            "metadata": metadata,
                        }
                    )

                elif result.result.type == "expired":
                    module_logger.warning(f"Request expired: {custom_id}")
                    responses.append(
                        {
                            "content": "Error: Request expired",
                            "generation_id": f"claude_batch_expired_{custom_id}",
                            "metadata": metadata,
                        }
                    )

                elif result.result.type == "canceled":
                    module_logger.warning(f"Request canceled: {custom_id}")
                    responses.append(
                        {
                            "content": "Error: Request canceled",
                            "generation_id": f"claude_batch_canceled_{custom_id}",
                            "metadata": metadata,
                        }
                    )

            module_logger.info(f"Successfully processed {len(responses)} batch responses")
            return responses

        except ImportError as e:
            module_logger.error(f"Anthropic SDK not available for batch processing: {e}")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)
        except Exception as e:
            module_logger.error(f"Claude batch API call failed: {e}")
            module_logger.exception("Batch processing error details:")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, temperature, **kwargs)


class LlmModel:
    """
    Unified LLM client that supports multiple providers (OpenRouter, OpenAI, Gemini, Claude).
    Routes requests to the appropriate provider based on model configuration.
    """

    DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_OPENROUTER_API_ENV = "OPENROUTER_API_KEY"
    DEFAULT_OPENAI_API_ENV = "OPENAI_API_KEY"
    DEFAULT_GEMINI_API_ENV = "GEMINI_API_KEY"
    DEFAULT_CLAUDE_API_ENV = "ANTHROPIC_API_KEY"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        model: ModelConfig,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        # OpenRouter settings
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        # Provider-specific API key environments
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
        claude_api_key_env: str = DEFAULT_CLAUDE_API_ENV,
        # Prompt logging control
        enable_prompt_logging: Optional[bool] = None,
    ) -> None:
        """
        Initialize LLM client with provider routing capabilities.

        Args:
            model: Model configuration (ModelConfig) specifying provider and model.
            temperature: Temperature for sampling.
            openrouter_api_key_env: Environment variable for OpenRouter API key.
            openrouter_base_url: OpenRouter base URL.
            openrouter_max_retries: Maximum retry attempts for OpenRouter.
            openrouter_retry_delay: Initial retry delay for OpenRouter.
            openrouter_backoff_factor: Backoff multiplier for OpenRouter.
            openai_api_key_env: Environment variable for OpenAI API key.
            gemini_api_key_env: Environment variable for Gemini API key.
            claude_api_key_env: Environment variable for Claude (Anthropic) API key.
            enable_prompt_logging: Whether to enable prompt content logging (defaults to False).
                If None, checks LLM_ENABLE_PROMPT_LOGGING environment variable.
        """
        self.model_config = model
        self.temperature = temperature

        # Determine prompt logging setting
        if enable_prompt_logging is None:
            # Check environment variable, default to False for security
            env_value = os.getenv("LLM_ENABLE_PROMPT_LOGGING", "false").lower()
            self.enable_prompt_logging = env_value in ("true", "1", "yes", "on")
        else:
            self.enable_prompt_logging = enable_prompt_logging

        # Initialize provider clients
        self._clients: dict[Provider, ProviderClient] = {}

        # Store configuration for lazy initialization
        # Explicit typing avoids mypy treating this as 'object'
        self._config: dict[str, dict[str, Any]] = {
            "openrouter": {
                "api_key_env": openrouter_api_key_env,
                "base_url": openrouter_base_url,
                "max_retries": openrouter_max_retries,
                "retry_delay": openrouter_retry_delay,
                "backoff_factor": openrouter_backoff_factor,
            },
            "openai": {"api_key_env": openai_api_key_env},
            "gemini": {"api_key_env": gemini_api_key_env},
            "claude": {"api_key_env": claude_api_key_env},
        }

        module_logger.info(
            f"Initializing LLM client: provider={self.model_config.provider.value}, model={self.model_config.model_id}"
        )

        # Validate that we have the required API key for the provider
        self._validate_api_key()

        module_logger.success(f"LLM client ready: {self.model_config.model_id}")

    def _validate_api_key(self) -> None:
        """Validate that the required API key is available for the provider."""
        provider = self.model_config.provider

        if provider == Provider.OPENROUTER:
            api_key_env = self._config["openrouter"]["api_key_env"]
        elif provider == Provider.OPENAI:
            api_key_env = self._config["openai"]["api_key_env"]
        elif provider == Provider.GEMINI:
            api_key_env = self._config["gemini"]["api_key_env"]
        elif provider == Provider.CLAUDE:
            api_key_env = self._config["claude"]["api_key_env"]
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not os.getenv(api_key_env):
            msg = f"Missing environment variable: {api_key_env}"
            module_logger.error(msg)
            raise ValueError(msg)

    def _get_client(self) -> ProviderClient:
        """Get or create the appropriate provider client."""
        provider = self.model_config.provider

        if provider not in self._clients:
            if provider == Provider.OPENROUTER:
                config = self._config["openrouter"]
                api_key = os.getenv(config["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {config['api_key_env']}")
                self._clients[provider] = OpenRouterClient(
                    api_key=api_key,
                    base_url=config["base_url"],
                    max_retries=config["max_retries"],
                    retry_delay=config["retry_delay"],
                    backoff_factor=config["backoff_factor"],
                )
            elif provider == Provider.OPENAI:
                api_key = os.getenv(self._config["openai"]["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {self._config['openai']['api_key_env']}")
                self._clients[provider] = OpenAIClient(api_key=api_key)
            elif provider == Provider.GEMINI:
                api_key = os.getenv(self._config["gemini"]["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {self._config['gemini']['api_key_env']}")
                self._clients[provider] = GeminiClient(api_key=api_key)
            elif provider == Provider.CLAUDE:
                api_key = os.getenv(self._config["claude"]["api_key_env"])
                if api_key is None:
                    raise ValueError(f"Missing environment variable: {self._config['claude']['api_key_env']}")
                self._clients[provider] = ClaudeClient(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return self._clients[provider]

    @classmethod
    def create(
        cls,
        model: ModelConfig,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        openrouter_api_key_env: str = DEFAULT_OPENROUTER_API_ENV,
        openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        openrouter_max_retries: int = DEFAULT_MAX_RETRIES,
        openrouter_retry_delay: float = DEFAULT_RETRY_DELAY,
        openrouter_backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        openai_api_key_env: str = DEFAULT_OPENAI_API_ENV,
        gemini_api_key_env: str = DEFAULT_GEMINI_API_ENV,
        claude_api_key_env: str = DEFAULT_CLAUDE_API_ENV,
        enable_prompt_logging: Optional[bool] = None,
    ) -> "LlmModel":
        """Create a new LlmModel instance with the specified configuration."""
        return cls(
            model=model,
            temperature=temperature,
            openrouter_api_key_env=openrouter_api_key_env,
            openrouter_base_url=openrouter_base_url,
            openrouter_max_retries=openrouter_max_retries,
            openrouter_retry_delay=openrouter_retry_delay,
            openrouter_backoff_factor=openrouter_backoff_factor,
            openai_api_key_env=openai_api_key_env,
            gemini_api_key_env=gemini_api_key_env,
            claude_api_key_env=claude_api_key_env,
            enable_prompt_logging=enable_prompt_logging,
        )

    @property
    def model_id(self) -> str:
        """Get the model ID for backward compatibility."""
        return self.model_config.model_id

    def __str__(self) -> str:
        return (
            f"LlmModel(provider={self.model_config.provider.value}, "
            f"model_id={self.model_config.model_id}, temperature={self.temperature})"
        )

    def __repr__(self) -> str:
        return (
            f"LlmModel(provider={self.model_config.provider.value}, "
            f"model_id={self.model_config.model_id}, temperature={self.temperature})"
        )

    def _log_prompt(self, role: str, content: str) -> None:
        """
        Logs a preview of the prompt content if prompt logging is enabled.

        Args:
            role (str): The role of the prompt (e.g., "System", "User").
            content (str): The full content of the prompt.
        """
        if self.enable_prompt_logging:
            if len(content) <= PROMPT_PREVIEW_MAX_LENGTH:
                preview = content
            else:
                preview = content[:PROMPT_PREVIEW_MAX_LENGTH] + "..."
            module_logger.trace(f"{role} prompt: {preview}")

    def supports_batch(self) -> bool:
        """
        Check if the current provider supports batch processing.

        Returns:
            bool: True if batch processing is supported, False otherwise
        """
        client = self._get_client()
        return client.supports_batch

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Make batch chat completion calls using the appropriate provider.

        Args:
            requests: List of request dictionaries
            temperature: Optional temperature override
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries

        Raises:
            LlmModelError: When batch API calls fail
            ValueError: If provider doesn't support batch processing
        """
        client = self._get_client()

        if not client.supports_batch:
            raise ValueError(f"Provider {self.model_config.provider.value} does not support batch processing")

        # Log batch info if enabled
        if self.enable_prompt_logging:
            module_logger.trace(f"Processing batch of {len(requests)} requests")

        # Ensure model_name is a concrete string
        model_name: str = self.model_config.provider_model_name or self.model_config.model_id

        # Use provided temperature or default
        call_temperature: float = temperature if temperature is not None else self.temperature

        return client.batch_chat_completion(
            requests=requests,
            model_name=model_name,
            temperature=call_temperature,
            **kwargs,
        )

    def close(self) -> None:
        """
        Close all provider clients and release resources (e.g., connection pools).

        This method iterates through all initialized provider clients and calls
        their close() method if available. This is important for proper cleanup
        of HTTP sessions and connection pools, especially for OpenRouter.
        """
        for provider, client in self._clients.items():
            try:
                if hasattr(client, "close") and callable(client.close):
                    client.close()
                    module_logger.debug(f"Closed {provider.value} client")
            except Exception as e:
                # Cleanup errors are non-critical but should be logged
                module_logger.debug(f"Error closing {provider.value} client: {e!r}")

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Make a chat completion call using the appropriate provider.

        Returns:
            Tuple of (assistant reply content, generation ID).

        Raises:
            LlmModelError: When API calls fail after max retries.
            ValueError: On empty/malformed response.
        """
        self._log_prompt(role="System", content=system_prompt)
        self._log_prompt(role="User", content=user_prompt)

        client = self._get_client()

        # Ensure model_name is a concrete string
        model_name: str = self.model_config.provider_model_name or self.model_config.model_id

        # Allow per-call temperature override without duplicating keyword
        call_temperature: float = kwargs.pop("temperature", self.temperature)

        return client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=call_temperature,
            **kwargs,
        )

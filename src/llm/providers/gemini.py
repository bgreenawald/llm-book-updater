import json
import time
from typing import Any, Tuple

from loguru import logger

from src.core.constants import OPENAI_BATCH_POLLING_INTERVAL
from src.llm.base import ProviderClient
from src.llm.cost_tracking import register_generation_model_info
from src.llm.exceptions import LlmModelError
from src.llm.utils import is_failed_response
from src.models.api_models import GeminiResponse

module_logger = logger


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

            # Explicitly set system_instruction, overriding any in kwargs
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            else:
                # Remove system_instruction if it exists in kwargs and no system_prompt provided
                config_kwargs.pop("system_instruction", None)

            # Create the config with deduplicated parameters
            config = types.GenerateContentConfig(**config_kwargs)

            response = client.models.generate_content(model=model_name, contents=contents, config=config)

            # Extract text from text parts, filtering out non-text parts like 'thought_signature'
            # This avoids duplication that can occur when response.text concatenates all parts incorrectly
            # The warning indicates response.text may concatenate text incorrectly when non-text parts exist
            content = ""
            text_parts = []
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content is not None
                    and hasattr(candidate.content, "parts")
                ):
                    parts = candidate.content.parts
                    if parts:
                        # Collect all text parts (skip non-text parts like thought_signature)
                        for part in parts:
                            if hasattr(part, "text") and part.text:
                                text_parts.append(part.text)
                            elif isinstance(part, dict) and part.get("text"):
                                text_parts.append(part["text"])

                        # Use only the first text part to avoid duplication
                        # If there are multiple text parts, they may be duplicates or the LLM may have
                        # split the response incorrectly
                        if text_parts:
                            if len(text_parts) > 1:
                                module_logger.warning(
                                    f"Found {len(text_parts)} text parts in Gemini response. "
                                    f"Part lengths: {[len(p) for p in text_parts]}. "
                                    f"Using only first part to avoid duplication. "
                                    f"If content appears truncated, consider concatenating all parts."
                                )
                            content = text_parts[0]

            # Fallback to response.text if explicit extraction didn't work
            if not content:
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                # Log a warning if we're falling back to response.text
                module_logger.warning("Falling back to response.text - explicit text extraction found no text parts")
                content = response.text

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

            return content, generation_id

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
        self,
        result_content: str,
        original_requests: list[dict[str, Any]],
        model_name: str,
    ) -> list[dict[str, Any]]:
        """
        Parse batch job results and match them with original requests.

        Args:
            result_content: JSONL content from batch job results
            original_requests: Original request list for metadata matching
            model_name: Name of the model used for the batch job

        Returns:
            List of response dictionaries
        """
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
                response = GeminiResponse.model_validate(response_data)

                # Get the generated content
                content = ""
                if response.candidates:
                    candidate = response.candidates[0]
                    content_parts = candidate.content.parts if candidate.content else []
                    if content_parts:
                        # Collect all text parts (skip non-text parts like thought_signature)
                        text_parts = []
                        for part in content_parts:
                            if isinstance(part, dict) and part.get("text"):
                                text_parts.append(part["text"])

                        # Use only the first text part to avoid duplication
                        # If there are multiple text parts, they may be duplicates or the LLM may have
                        # split the response incorrectly
                        if text_parts:
                            if len(text_parts) > 1:
                                module_logger.warning(
                                    f"Found {len(text_parts)} text parts in Gemini batch response. "
                                    f"Part lengths: {[len(p) for p in text_parts]}. "
                                    f"Using only first part to avoid duplication. "
                                    f"If content appears truncated, consider concatenating all parts."
                                )
                            content = text_parts[0]

                # Generate a unique ID
                generation_id = f"gemini_batch_{int(time.time())}_{key}"

                # Get original metadata
                metadata = key_to_metadata.get(key, {})

                # Track token usage if available
                try:
                    usage_metadata = response.usage_metadata
                    if usage_metadata:
                        prompt_tokens = usage_metadata.prompt_token_count
                        completion_tokens = usage_metadata.candidates_token_count

                        if prompt_tokens is not None or completion_tokens is not None:
                            register_generation_model_info(
                                generation_id=generation_id,
                                model=model_name,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                is_batch=True,  # This is batch processing, apply 50% discount
                            )
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    # Usage tracking is optional for batch; log but don't fail
                    module_logger.debug(f"Failed to track Gemini batch usage for {generation_id}: {e}")

                # Check if response is a failure (empty content)
                response_failed = is_failed_response(content)
                responses.append(
                    {
                        "content": content,
                        "generation_id": generation_id,
                        "metadata": metadata,
                        "failed": response_failed,
                    }
                )

            except (json.JSONDecodeError, KeyError, ValueError, IndexError, AttributeError) as e:
                module_logger.error(f"Error parsing batch result line: {e}")
                # Add error response
                responses.append(
                    {
                        "content": f"Error parsing result: {str(e)}",
                        "generation_id": f"gemini_batch_parse_error_{int(time.time())}",
                        "metadata": {},
                        "failed": True,
                    }
                )

        return responses

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
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
                    return super().batch_chat_completion(requests, model_name, **kwargs)

                # Get the completed job to retrieve results
                completed_job = client.batches.get(name=batch_job.name)

                # Download results
                if hasattr(completed_job, "dest") and hasattr(completed_job.dest, "file_name"):
                    result_file_name = completed_job.dest.file_name
                    module_logger.info(f"Downloading batch results from {result_file_name}...")

                    result_bytes = client.files.download(file=result_file_name)
                    result_content = result_bytes.decode("utf-8")

                    # Parse and return results
                    responses = self._parse_batch_results(result_content, requests, model_name)

                    module_logger.info(f"Successfully processed {len(responses)} batch responses")
                    return responses
                else:
                    module_logger.error("No result file found in completed batch job")
                    # Fall back to sequential processing
                    return super().batch_chat_completion(requests, model_name, **kwargs)

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
            return super().batch_chat_completion(requests, model_name, **kwargs)
        except Exception as e:
            module_logger.error(f"Gemini batch API call failed: {e}")
            module_logger.exception("Batch processing error details:")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, **kwargs)

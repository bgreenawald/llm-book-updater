import json
import os
import tempfile
import time
from typing import Any, Dict, Tuple

from loguru import logger

from src.core.constants import (
    OPENAI_BATCH_DEFAULT_TIMEOUT,
    OPENAI_BATCH_POLLING_INTERVAL,
)
from src.llm.base import ProviderClient
from src.llm.cost_tracking import register_generation_model_info
from src.llm.exceptions import LlmModelError, ResponseTruncatedError
from src.llm.utils import is_failed_response

module_logger = logger


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
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using OpenAI SDK."""
        client = self._get_client()

        try:
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "instructions": system_prompt,
                "input": user_prompt,
            }

            # Handle reasoning parameter - support both formats:
            # 1. reasoning={"effort": "high"} (standard format)
            # 2. effort="high" (legacy format, will be wrapped)
            reasoning = kwargs.pop("reasoning", None)
            effort = kwargs.pop("effort", None)

            if reasoning is not None:
                if isinstance(reasoning, dict):
                    request_kwargs["reasoning"] = reasoning
                else:
                    module_logger.warning(
                        f"Invalid reasoning parameter type: {type(reasoning)}. "
                        f"Expected dict with 'effort' key, e.g., {{'effort': 'high'}}"
                    )
            elif effort is not None:
                # Legacy format: wrap effort in reasoning dict
                request_kwargs["reasoning"] = {"effort": effort}

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
                raise ResponseTruncatedError(
                    message="Response truncated due to max_tokens limit",
                    model_name=model_name,
                )

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

    def _create_batch_jsonl(self, requests: list[dict[str, Any]], model_name: str, **kwargs) -> str:
        """
        Create JSONL content for batch processing using the OpenAI Responses API.

        Args:
            requests: List of request dictionaries containing system_prompt, user_prompt, metadata
            model_name: Name of the model to use
            **kwargs: Additional parameters (e.g., effort under reasoning)

        Returns:
            str: JSONL formatted string for batch processing
        """
        jsonl_lines = []
        for i, request in enumerate(requests):
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("user_prompt", "")
            metadata = request.get("metadata", {})

            # Build request body for Responses API
            batch_body: Dict[str, Any] = {
                "model": model_name,
                "input": user_prompt,
            }

            if system_prompt:
                batch_body["instructions"] = system_prompt

            # Handle reasoning parameter - support both formats:
            # 1. reasoning={"effort": "high"} (standard format)
            # 2. effort="high" (legacy format, will be wrapped)
            reasoning = kwargs.get("reasoning")
            effort = kwargs.get("effort")

            if reasoning is not None:
                if isinstance(reasoning, dict):
                    batch_body["reasoning"] = reasoning
                else:
                    module_logger.warning(
                        f"Invalid reasoning parameter type in batch request: {type(reasoning)}. "
                        f"Expected dict with 'effort' key, e.g., {{'effort': 'high'}}"
                    )
            elif effort is not None:
                # Legacy format: wrap effort in reasoning dict
                batch_body["reasoning"] = {"effort": effort}

            # Pass through any additional kwargs
            batch_body.update(kwargs)

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
        self,
        result_content: str,
        original_requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Parse batch job results and match them with original requests.

        Args:
            result_content: JSONL content from batch job results
            original_requests: Original request list for metadata matching

        Returns:
            List of response dictionaries
        """
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
                            "failed": True,
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

            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                module_logger.error(f"Error parsing batch result line: {e}")
                responses.append(
                    {
                        "content": f"Error parsing result: {str(e)}",
                        "generation_id": f"openai_batch_parse_error_{int(time.time())}",
                        "metadata": {},
                        "failed": True,
                    }
                )

        return responses

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
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
            batch_timeout: Maximum time to wait for batch job completion (seconds)
            **kwargs: Additional arguments

        Returns:
            List of response dictionaries
        """
        client = self._get_client()

        try:
            module_logger.info(f"Starting OpenAI batch processing for {len(requests)} requests")

            # Create JSONL content for batch processing
            jsonl_content = self._create_batch_jsonl(requests, model_name, **kwargs)

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
                    return super().batch_chat_completion(requests, model_name, **kwargs)

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
                    return super().batch_chat_completion(requests, model_name, **kwargs)

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
            return super().batch_chat_completion(requests, model_name, **kwargs)
        except Exception as e:
            module_logger.error(f"OpenAI batch API call failed: {e}")
            module_logger.exception("Batch processing error details:")
            # Fall back to sequential processing
            return super().batch_chat_completion(requests, model_name, **kwargs)

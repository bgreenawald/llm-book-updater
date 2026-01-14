"""Gemini provider client with batch support."""

import json
import time
import uuid
from typing import Any, Tuple

from loguru import logger

from llm_core.api_models import GeminiResponse
from llm_core.config import OPENAI_BATCH_POLLING_INTERVAL
from llm_core.cost import register_generation_model_info
from llm_core.exceptions import LlmModelError
from llm_core.providers.base import ProviderClient
from llm_core.utils import is_failed_response

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

    def _extract_text_from_parts(self, parts: list) -> str:
        """Extract text from response parts."""
        text_parts = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            elif isinstance(part, dict) and part.get("text"):
                text_parts.append(part["text"])

        if text_parts:
            if len(text_parts) > 1:
                module_logger.warning(f"Found {len(text_parts)} text parts in Gemini response. Using only first part.")
            return text_parts[0]
        return ""

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

            # Remove unsupported parameters
            kwargs.pop("reasoning", None)

            contents = kwargs.pop("contents", user_prompt)
            config_kwargs = dict(kwargs)

            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            else:
                config_kwargs.pop("system_instruction", None)

            config = types.GenerateContentConfig(**config_kwargs)
            response = client.models.generate_content(model=model_name, contents=contents, config=config)

            content = ""
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                    parts = candidate.content.parts
                    if parts:
                        content = self._extract_text_from_parts(parts)

            if not content:
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                content = response.text

            generation_id = f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}"

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
                module_logger.debug(f"Could not extract Gemini usage metadata: {e}")

            return content, generation_id

        except Exception as e:
            module_logger.error(f"Gemini API call failed: {e}")
            raise LlmModelError(f"Gemini API call failed: {e}") from e

    def _create_batch_jsonl(self, requests: list[dict[str, Any]]) -> str:
        """Create JSONL content for batch processing."""
        jsonl_lines = []
        for i, request in enumerate(requests):
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("user_prompt", "")
            metadata = request.get("metadata", {})

            user_parts = []
            if user_prompt:
                user_parts.append({"text": user_prompt})

            batch_request = {
                "key": f"request_{i}",
                "request": {"contents": [{"parts": user_parts}]},
                "metadata": metadata,
            }

            if system_prompt:
                batch_request["request"]["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            jsonl_lines.append(json.dumps(batch_request))

        return "\n".join(jsonl_lines)

    def _retry_network_call(self, func, max_retries: int = 3, retry_delay: int = 30):
        """Retry a network call with exponential backoff for OSError."""
        last_error: OSError | None = None
        for attempt in range(max_retries):
            try:
                return func()
            except OSError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    module_logger.warning(
                        f"Network error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
        if last_error is not None:
            raise last_error
        raise OSError("Network call failed with no error captured")

    def _poll_batch_job(self, client, job_name: str, timeout_seconds: int = 3600 * 24) -> bool:
        """Poll batch job until completion or timeout."""
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
                module_logger.error(f"Error polling batch job: {e}")
                return False
            except OSError as e:
                module_logger.warning(f"Network error polling batch job, will retry: {e}")
                time.sleep(poll_interval)
                continue

        module_logger.error(f"Batch job timed out after {timeout_seconds} seconds")
        return False

    def _parse_batch_results(
        self,
        result_content: str,
        original_requests: list[dict[str, Any]],
        model_name: str,
    ) -> list[dict[str, Any]]:
        """Parse batch job results."""
        responses = []
        result_lines = result_content.strip().split("\n")

        key_to_metadata = {}
        for i, request in enumerate(original_requests):
            key_to_metadata[f"request_{i}"] = request.get("metadata", {})

        for line in result_lines:
            try:
                result = json.loads(line)
                key = result.get("key", "")
                response_data = result.get("response", {})
                response = GeminiResponse.model_validate(response_data)

                content = ""
                if response.candidates:
                    candidate = response.candidates[0]
                    content_parts = candidate.content.parts if candidate.content else []
                    if content_parts:
                        content = self._extract_text_from_parts(content_parts)

                generation_id = f"gemini_batch_{int(time.time())}_{uuid.uuid4().hex[:8]}_{key}"
                metadata = key_to_metadata.get(key, {})

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
                                is_batch=True,
                            )
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    module_logger.debug(f"Failed to track Gemini batch usage: {e}")

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
        """Process batch chat completion requests using Gemini's batch API."""
        client = self._get_client()

        try:
            import os
            import tempfile

            from google.genai.types import CreateBatchJobConfig

            kwargs.pop("reasoning", None)

            module_logger.info(f"Starting Gemini batch processing for {len(requests)} requests")

            jsonl_content = self._create_batch_jsonl(requests)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
                temp_file.write(jsonl_content)
                temp_file_path = temp_file.name

            try:
                module_logger.info("Uploading batch requests file...")
                uploaded_file = client.files.upload(file=temp_file_path, config={"mimeType": "application/jsonl"})

                batch_config = CreateBatchJobConfig(
                    display_name=f"llm_core_batch_{int(time.time())}",
                )

                module_logger.info(f"Creating batch job with model {model_name}...")
                batch_job = client.batches.create(
                    model=model_name,
                    src=uploaded_file.name,
                    config=batch_config,
                )

                module_logger.info(f"Created batch job: {batch_job.name}")

                if not self._poll_batch_job(client, batch_job.name, batch_timeout):
                    return super().batch_chat_completion(requests, model_name, **kwargs)

                completed_job = self._retry_network_call(lambda: client.batches.get(name=batch_job.name))

                if hasattr(completed_job, "dest") and hasattr(completed_job.dest, "file_name"):
                    result_file_name = completed_job.dest.file_name
                    result_bytes = self._retry_network_call(lambda: client.files.download(file=result_file_name))
                    result_content = result_bytes.decode("utf-8")

                    responses = self._parse_batch_results(result_content, requests, model_name)
                    module_logger.info(f"Successfully processed {len(responses)} batch responses")
                    return responses
                else:
                    return super().batch_chat_completion(requests, model_name, **kwargs)

            finally:
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except ImportError as e:
            module_logger.error(f"Google GenAI SDK not available: {e}")
            return super().batch_chat_completion(requests, model_name, **kwargs)
        except Exception as e:
            module_logger.error(f"Gemini batch API call failed: {e}")
            return super().batch_chat_completion(requests, model_name, **kwargs)

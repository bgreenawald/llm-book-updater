"""Claude (Anthropic) provider client with batch support."""

from typing import Any, Tuple

from loguru import logger

from llm_core.cost import register_generation_model_info
from llm_core.exceptions import LlmModelError, ResponseTruncatedError
from llm_core.providers.base import ProviderClient
from llm_core.utils import is_failed_response

module_logger = logger


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
        **kwargs,
    ) -> Tuple[str, str]:
        """Make a chat completion call using Anthropic SDK."""
        client = self._get_client()

        try:
            request_params: dict[str, Any] = {
                "model": model_name,
                "max_tokens": kwargs.pop("max_tokens", 65536),
                "messages": [{"role": "user", "content": user_prompt}],
            }

            if system_prompt:
                request_params["system"] = system_prompt

            request_params.update(kwargs)

            response = client.messages.create(**request_params)

            if not response.content or not response.content[0].text:
                raise ValueError("Empty response from Claude")

            content = response.content[0].text

            if response.stop_reason == "max_tokens":
                module_logger.warning("Response truncated: consider increasing max_tokens")
                raise ResponseTruncatedError(
                    message="Response truncated due to max_tokens limit",
                    model_name=model_name,
                )

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
                module_logger.debug(f"Failed to extract usage metadata for {model_name}: {e}")

            return content, response.id

        except ResponseTruncatedError:
            raise
        except Exception as e:
            module_logger.error(f"Claude API call failed: {e}")
            raise LlmModelError(f"Claude API call failed: {e}") from e

    def batch_chat_completion(
        self,
        requests: list[dict[str, Any]],
        model_name: str,
        batch_timeout: int = 3600 * 24,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Process batch chat completion requests using Claude's Message Batches API."""
        client = self._get_client()

        try:
            import time

            module_logger.info(f"Starting Claude batch processing for {len(requests)} requests")

            batch_requests = []
            for i, request in enumerate(requests):
                system_prompt = request.get("system_prompt", "")
                user_prompt = request.get("user_prompt", "")
                metadata = request.get("metadata", {})

                params: dict[str, Any] = {
                    "model": model_name,
                    "max_tokens": kwargs.get("max_tokens", 65536),
                    "messages": [{"role": "user", "content": user_prompt}],
                }

                if system_prompt:
                    params["system"] = system_prompt

                params.update(kwargs)

                batch_request = {
                    "custom_id": f"request_{i}_{metadata.get('id', '')}",
                    "params": params,
                }

                batch_requests.append(batch_request)

            module_logger.info(f"Creating Claude batch job with model {model_name}...")
            message_batch = client.messages.batches.create(requests=batch_requests)

            module_logger.info(f"Created batch job: {message_batch.id}")

            module_logger.info(f"Polling batch job for completion (timeout: {batch_timeout}s)...")
            start_time = time.time()
            poll_interval = 60

            while time.time() - start_time < batch_timeout:
                message_batch = client.messages.batches.retrieve(message_batch.id)

                module_logger.info(f"Batch job {message_batch.id} status: {message_batch.processing_status}")

                if message_batch.processing_status == "ended":
                    break
                elif message_batch.processing_status in ["canceling", "canceled"]:
                    module_logger.error("Batch job was canceled")
                    return super().batch_chat_completion(requests, model_name, **kwargs)

                time.sleep(poll_interval)
            else:
                module_logger.error(f"Batch job timed out after {batch_timeout} seconds")
                return super().batch_chat_completion(requests, model_name, **kwargs)

            module_logger.info("Downloading batch results...")

            id_to_metadata = {}
            for i, request in enumerate(requests):
                metadata = request.get("metadata", {})
                custom_id = f"request_{i}_{metadata.get('id', '')}"
                id_to_metadata[custom_id] = metadata

            responses = []
            for result in client.messages.batches.results(message_batch.id):
                custom_id = result.custom_id
                metadata = id_to_metadata.get(custom_id, {})

                if result.result.type == "succeeded":
                    message = result.result.message
                    content = message.content[0].text if message.content else ""
                    generation_id = message.id

                    try:
                        usage = message.usage
                        if usage:
                            register_generation_model_info(
                                generation_id=generation_id,
                                model=model_name,
                                prompt_tokens=usage.input_tokens,
                                completion_tokens=usage.output_tokens,
                                is_batch=True,
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        module_logger.debug(f"Failed to track batch usage for {generation_id}: {e}")

                    response_failed = is_failed_response(content)
                    responses.append(
                        {
                            "content": content,
                            "generation_id": generation_id,
                            "metadata": metadata,
                            "failed": response_failed,
                        }
                    )

                elif result.result.type == "errored":
                    error = result.result.error
                    error_message = error.message if hasattr(error, "message") else str(error)
                    module_logger.error(f"Error in batch result {custom_id}: {error_message}")
                    responses.append(
                        {
                            "content": f"Error: {error_message}",
                            "generation_id": f"claude_batch_error_{custom_id}",
                            "metadata": metadata,
                            "failed": True,
                        }
                    )

                elif result.result.type == "expired":
                    responses.append(
                        {
                            "content": "Error: Request expired",
                            "generation_id": f"claude_batch_expired_{custom_id}",
                            "metadata": metadata,
                            "failed": True,
                        }
                    )

                elif result.result.type == "canceled":
                    responses.append(
                        {
                            "content": "Error: Request canceled",
                            "generation_id": f"claude_batch_canceled_{custom_id}",
                            "metadata": metadata,
                            "failed": True,
                        }
                    )

            module_logger.info(f"Successfully processed {len(responses)} batch responses")
            return responses

        except ImportError as e:
            module_logger.error(f"Anthropic SDK not available for batch processing: {e}")
            return super().batch_chat_completion(requests, model_name, **kwargs)
        except Exception as e:
            module_logger.error(f"Claude batch API call failed: {e}")
            return super().batch_chat_completion(requests, model_name, **kwargs)

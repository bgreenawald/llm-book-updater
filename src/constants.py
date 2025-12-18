"""
Constants for the LLM Book Updater application.

This module centralizes all magic numbers and configuration constants used throughout
the application for better maintainability and consistency.
"""

# =============================================================================
# API Configuration - Timeouts & Polling
# =============================================================================

# OpenRouter API configuration
OPENROUTER_REQUEST_TIMEOUT = 30  # seconds
OPENROUTER_API_TIMEOUT = 10  # seconds
DEFAULT_OPENROUTER_MAX_RETRIES = 3
DEFAULT_OPENROUTER_RETRY_DELAY = 1.0  # seconds
DEFAULT_OPENROUTER_BACKOFF_FACTOR = 2.0

# OpenAI Batch API configuration
OPENAI_BATCH_POLLING_INTERVAL = 120  # seconds (2 minutes)
OPENAI_BATCH_DEFAULT_TIMEOUT = 3600 * 24  # seconds (24 hours)

# OpenRouter Models API timeout
OPENROUTER_MODELS_API_TIMEOUT = 30  # seconds

# OpenRouter connection pooling configuration
OPENROUTER_POOL_CONNECTIONS = 10  # Number of connection pools (one per host)
OPENROUTER_POOL_MAXSIZE = 20  # Maximum connections per pool


# =============================================================================
# Processing Configuration
# =============================================================================

# Batch processing discount rate (50% off for batch API usage)
BATCH_PROCESSING_DISCOUNT_RATE = 0.5

# Default number of parallel workers for concurrent processing
DEFAULT_MAX_WORKERS = 1

# Default maximum retries for failed LLM generations (when retry is enabled)
DEFAULT_GENERATION_MAX_RETRIES = 2

# Default length reduction bounds (percentage) for text compression
DEFAULT_LENGTH_REDUCTION_BOUNDS = (35, 50)

# Input file index prefix in output directory
INPUT_FILE_INDEX_PREFIX = "00"

# Sub-block processing defaults
DEFAULT_MAX_SUBBLOCK_TOKENS = 4096
DEFAULT_MIN_SUBBLOCK_TOKENS = 1024
MIN_SUBBLOCK_TOKEN_BOUND = 256
MAX_SUBBLOCK_TOKEN_BOUND = 32000


# =============================================================================
# LLM Model Defaults
# =============================================================================

# Default temperature for LLM requests
LLM_DEFAULT_TEMPERATURE = 1
GEMINI_DEFAULT_TEMPERATURE = 1


# =============================================================================
# Content Processing
# =============================================================================

# Maximum length for prompt preview in logs
PROMPT_PREVIEW_MAX_LENGTH = 200

# Default tags to preserve during processing
DEFAULT_TAGS_TO_PRESERVE = ["{preface}", "{license}"]

# Markdown header levels
MARKDOWN_HEADER_MIN_LEVEL = 1
MARKDOWN_HEADER_MAX_LEVEL = 6


# =============================================================================
# Cost Calculation
# =============================================================================

# Base unit for token pricing (1 million tokens)
TOKENS_PER_MILLION = 1_000_000


# =============================================================================
# Model Pricing - OpenAI (USD per 1M tokens)
# =============================================================================

# o4-mini pricing
OPENAI_O4_MINI_INPUT_PRICE_PER_1M = 1.10
OPENAI_O4_MINI_OUTPUT_PRICE_PER_1M = 4.40

# GPT-4o pricing
OPENAI_GPT4O_INPUT_PRICE_PER_1M = 2.50
OPENAI_GPT4O_OUTPUT_PRICE_PER_1M = 10.00

# GPT-4o-mini pricing
OPENAI_GPT4O_MINI_INPUT_PRICE_PER_1M = 0.15
OPENAI_GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.60

# GPT-3.5-turbo pricing (legacy)
OPENAI_GPT35_INPUT_PRICE_PER_1M = 0.50
OPENAI_GPT35_OUTPUT_PRICE_PER_1M = 1.50


# =============================================================================
# Model Pricing - Gemini (USD per 1M tokens)
# =============================================================================

# Gemini 2.5 Flash pricing
GEMINI_FLASH_INPUT_PRICE_PER_1M = 0.30
GEMINI_FLASH_OUTPUT_PRICE_PER_1M = 2.50

# Gemini 2.5 Pro pricing
GEMINI_PRO_INPUT_PRICE_PER_1M = 1.25
GEMINI_PRO_OUTPUT_PRICE_PER_1M = 10.00

# Gemini 2.5 Flash Lite pricing
GEMINI_FLASH_LITE_INPUT_PRICE_PER_1M = 0.10
GEMINI_FLASH_LITE_OUTPUT_PRICE_PER_1M = 0.40

# Cost Tracking Documentation

This document explains the cost tracking functionality implemented in the LLM Book Updater project, which allows you to monitor and log costs for OpenRouter API usage across pipeline phases.

## Overview

The cost tracking system provides detailed monitoring of API usage and costs by:

1. **Capturing Generation IDs**: Each API call returns a generation ID that can be used to query detailed statistics
2. **Querying Cost Data**: Using OpenRouter's `/api/v1/generation` endpoint to get precise token counts and costs
3. **Aggregating by Phase**: Calculating costs per phase and total run costs
4. **Detailed Logging**: Providing comprehensive cost breakdowns in logs and metadata files

## How It Works

### 1. Generation ID Capture

When the LLM model makes an API call, the response includes a `generation_id` field. The system captures this ID and stores it for each phase:

```python
# In LlmModel.chat_completion() (after patching)
content, generation_id = self.model.chat_completion(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)
add_generation_id(phase_name=self.name, generation_id=generation_id)
```

### 2. Cost Calculation

After each phase completes, the system queries OpenRouter's generation stats endpoint to get detailed information:

```python
# Query generation statistics
stats = cost_tracker.get_generation_stats(generation_id)
# Returns: GenerationStats with tokens, cost, model info, etc.
```

### 3. Phase and Run Aggregation

Costs are aggregated at multiple levels:

- **Per Generation**: Individual API call costs
- **Per Phase**: Total costs for all generations in a phase
- **Per Run**: Total costs across all phases

## Usage

### Option 1: Using the Cost Tracking Wrapper (Recommended)

The cost tracking wrapper provides a non-intrusive way to add cost tracking without modifying existing code:

```python
from src.cost_tracking_wrapper import add_generation_id, calculate_and_log_costs

# After each API call, add the generation ID
processed_body, generation_id = self.model.chat_completion(...)
add_generation_id(phase_name=self.name, generation_id=generation_id)

# At the end of your pipeline, calculate and log costs
phase_names = ["modernize", "edit", "final"]
run_costs = calculate_and_log_costs(phase_names)
```

### Option 2: Manual Cost Tracking

You can also manually create a cost tracker and manage generation IDs:

```python
from src.cost_tracker import CostTracker

# Create cost tracker
cost_tracker = CostTracker(api_key="your-api-key")

# Store generation IDs manually
generation_ids = ["gen_123", "gen_456", "gen_789"]

# Calculate phase costs
phase_cost = cost_tracker.calculate_phase_costs(
    phase_name="modernize",
    phase_index=0,
    generation_ids=generation_ids,
)

# Calculate total run costs
run_costs = cost_tracker.calculate_run_costs([phase_cost])
cost_tracker.log_detailed_costs(run_costs)
```

## Implementation Steps

### Step 1: Patch the LlmModel (Optional)

To capture generation IDs automatically, you can patch the `LlmModel.chat_completion` method:

1. **Add import** to `src/llm_model.py`:
   ```python
   from typing import Tuple
   ```

2. **Change return type**:
   ```python
   def chat_completion(
       self,
       system_prompt: str,
       user_prompt: str,
       **kwargs,
   ) -> Tuple[str, str]:  # Changed from str to Tuple[str, str]
   ```

3. **Extract generation ID**:
   ```python
   content = choices[0]["message"]["content"]
   finish_reason = choices[0].get("finish_reason", "unknown")
   generation_id = resp_data.get("id", "unknown")  # Add this line
   ```

4. **Return tuple**:
   ```python
   return content, generation_id  # Return tuple instead of just content
   ```

### Step 2: Update Calling Code

Update all places where `chat_completion` is called:

```python
# Old code:
processed_body = self.model.chat_completion(...)

# New code:
processed_body, generation_id = self.model.chat_completion(...)
add_generation_id(phase_name=self.name, generation_id=generation_id)
```

### Step 3: Add Cost Calculation

Add cost calculation at the end of your pipeline:

```python
# After pipeline completion
phase_names = [phase.name for phase in config.phases if phase.enabled]
run_costs = calculate_and_log_costs(phase_names)
```

## Output

### Console Logging

Cost information is logged to the console during and after pipeline execution:

```
Phase modernize costs: 1,234 tokens (567 prompt, 667 completion), $0.002345 USD, 3 generations
Total run costs: 1,234 tokens (567 prompt, 667 completion), $0.002345 USD, 3 generations across 1 phases
```

### Detailed Cost Breakdown

At the end of the run, a detailed breakdown is logged:

```
================================================================================
DETAILED COST BREAKDOWN
================================================================================
Phase 1: modernize
  Tokens: 1,234 (567 prompt, 667 completion)
  Cost: $0.002345 USD
  Generations: 3

================================================================================
TOTAL RUN SUMMARY
================================================================================
Total Phases: 1
Completed Phases: 1
Total Generations: 3
Total Tokens: 1,234 (567 prompt, 667 completion)
Total Cost: $0.002345 USD
================================================================================
```

### Metadata Files

Two types of metadata files are created:

#### 1. Pipeline Metadata (`pipeline_metadata_*.json`)

Contains comprehensive information about the pipeline run, including generation counts per phase.

#### 2. Cost Metadata (`cost_metadata_*.json`)

Contains detailed cost information:

```json
{
  "run_timestamp": "2024-01-15T10:30:00",
  "book_name": "Example Book",
  "author_name": "Example Author",
  "total_phases": 1,
  "completed_phases": 1,
  "total_generations": 3,
  "total_prompt_tokens": 567,
  "total_completion_tokens": 667,
  "total_tokens": 1234,
  "total_cost": 0.002345,
  "currency": "USD",
  "phase_costs": [
    {
      "phase_name": "modernize",
      "phase_index": 0,
      "generation_count": 3,
      "total_prompt_tokens": 567,
      "total_completion_tokens": 667,
      "total_tokens": 1234,
      "total_cost": 0.002345,
      "currency": "USD"
    }
  ]
}
```

## API Reference

### CostTracker Class

Main class for tracking costs and querying generation statistics.

#### Methods

- `get_generation_stats(generation_id: str) -> Optional[GenerationStats]`
  - Query detailed statistics for a specific generation
  - Returns None if the generation cannot be retrieved

- `calculate_phase_costs(phase_name: str, phase_index: int, generation_ids: List[str]) -> PhaseCosts`
  - Calculate total costs for a specific phase
  - Aggregates costs from all generations in the phase

- `calculate_run_costs(phase_costs: List[PhaseCosts]) -> RunCosts`
  - Calculate total costs for the entire run
  - Aggregates costs from all phases

- `log_detailed_costs(run_costs: RunCosts) -> None`
  - Log a detailed cost breakdown to the console

### CostTrackingWrapper Class

Wrapper class that provides easy access to cost tracking functionality.

#### Methods

- `add_generation_id(phase_name: str, generation_id: str) -> None`
  - Add a generation ID for a specific phase

- `calculate_and_log_costs(phase_names: List[str]) -> Optional[RunCosts]`
  - Calculate and log costs for all tracked phases

### Data Classes

#### GenerationStats

```python
@dataclass
class GenerationStats:
    generation_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    currency: str = "USD"
    created_at: Optional[str] = None
    finish_reason: Optional[str] = None
```

#### PhaseCosts

```python
@dataclass
class PhaseCosts:
    phase_name: str
    phase_index: int
    generation_ids: List[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost: float
    currency: str = "USD"
    generation_count: int = 0
```

#### RunCosts

```python
@dataclass
class RunCosts:
    total_phases: int
    completed_phases: int
    total_generations: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost: float
    phase_costs: List[PhaseCosts]
    currency: str = "USD"
```

## Example Usage

See `examples/cost_tracking_example.py` for a complete example of how to use the cost tracking functionality.

## Notes

### Token Counting

- Token counts are based on the **native** tokenizer for each model (not normalized counts)
- This ensures accurate cost calculation based on the actual pricing for each model
- The system caches generation statistics to avoid duplicate API calls

### Error Handling

- If cost tracking fails to initialize (e.g., no API key), the pipeline continues without cost tracking
- Failed generation queries are logged as warnings but don't stop the pipeline
- Cost calculations gracefully handle missing or invalid generation data

### Performance

- Generation statistics are cached to avoid duplicate API calls
- Cost calculations are performed after all phases complete to minimize impact on pipeline performance
- The system uses efficient aggregation to handle large numbers of generations

## Troubleshooting

### No Cost Information

If you don't see cost information:

1. Check that `OPENROUTER_API_KEY` environment variable is set
2. Verify the API key is valid and has sufficient permissions
3. Check the logs for any cost tracking initialization errors

### Missing Generation Data

If some generations show as missing:

1. Check network connectivity to OpenRouter API
2. Verify the generation IDs are valid
3. Check if the generations are still available (they may expire after some time)

### Inaccurate Costs

If costs seem inaccurate:

1. Verify you're using the correct model pricing
2. Check that token counts match your expectations
3. Ensure you're looking at the correct currency 
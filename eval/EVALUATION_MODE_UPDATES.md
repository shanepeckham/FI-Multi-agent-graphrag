# Evaluation Mode Field Updates

## Summary

Added the `evaluation_mode` field to support better evaluation testing by disabling WebSocket updates during evaluation runs.

## Files Updated

### 1. API Request Models ✅
- `/Users/shanepeckham/sources/graphrag/app/main.py` - QueryRequest model already contained the field

### 2. API Documentation ✅
- `/Users/shanepeckham/sources/graphrag/app/main.py` - Updated root endpoint to include evaluation_mode parameter info

### 3. Main README ✅
- `/Users/shanepeckham/sources/graphrag/README.md` - Added evaluation_mode to API example and parameters section

### 4. Evaluation Scripts ✅
- `/Users/shanepeckham/sources/graphrag/eval/evaluate_johnson_api.py` - Added evaluation_mode: true to API payload
- `/Users/shanepeckham/sources/graphrag/eval/test_single_question.py` - Added evaluation_mode: true to API payload

### 5. Evaluation README ✅
- `/Users/shanepeckham/sources/graphrag/eval/README.md` - Added dedicated section explaining evaluation mode

## Field Definition

```json
{
  "evaluation_mode": {
    "type": "boolean",
    "default": false,
    "description": "Whether to run in evaluation mode (no WebSocket updates)"
  }
}
```

## Benefits

- Prevents WebSocket interference during evaluation
- Reduces system overhead for testing
- Ensures consistent evaluation performance
- Makes evaluation results more reliable

## Usage

The evaluation scripts automatically set this to `true`, while normal API usage defaults to `false` to enable real-time dashboard updates.

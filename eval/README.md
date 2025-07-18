# Johnson & Johnson Financial Analysis Evaluation

This directory contains evaluation scripts for testing the Agent Team's performance on Johnson & Johnson financial analysis questions.

## Files

- `johnson.jsonl` - Contains financial analysis questions from the FinanceBench dataset
- `evaluate_johnson_api.py` - Evaluation script that calls the `/query_team` API endpoint
- `evaluate_johnson.py` - Direct evaluation script that imports agent modules (may need setup)
- `test_setup.py` - Test script to verify the evaluation setup
- `test_single_question.py` - Quick test script to run a single question
- `requirements.txt` - Python dependencies for the evaluation scripts
- `evaluation_results.json` - Output file containing evaluation results (generated after running)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your environment variables are set up in `../apple/.env`:
```
AZURE_OPENAI_API_KEY=your_api_key_here
API_KEY=your_api_key_here  # For API authentication
API_BASE_URL=http://localhost:8000  # Optional, defaults to localhost:8000
```

## Usage

### Quick Test

First, verify your setup works:
```bash
python test_setup.py
```

Test a single question:
```bash
# Start the API server first
cd ../app
python main.py

# In another terminal, run the single question test
cd ../eval
python test_single_question.py
```

### Method 1: API-based Evaluation (Recommended)

First, make sure the GraphRAG API server is running:
```bash
cd ../app
python main.py
```

Then run the evaluation:
```bash
python evaluate_johnson_api.py
```

### Method 2: Direct Module Evaluation

```bash
python evaluate_johnson.py
```

## Output

The evaluation script will:
1. Load questions from `johnson.jsonl`
2. Process each question through the agent team
3. Save results to `evaluation_results.json`
4. Print a summary of the evaluation

## Sample Output

```
Starting Johnson & Johnson Financial Analysis Evaluation
================================================================================
API Base URL: http://localhost:8000
API Key configured: Yes
✅ API connection successful: {'status': 'healthy', 'service': 'GraphRAG API'}
Loaded 10 questions from johnson.jsonl

Processing question 1/10

================================================================================
Evaluating Question ID: financebench_id_00956
Company: Johnson & Johnson
Question: Are JnJ's FY2022 financials that of a high growth company?
Expected Answer: No, JnJ's FY2022 financials are not of a high growth company as sales grew by 1.3% in FY2022.
================================================================================

Agent Response:
# Agent Team Analysis Report
...

Response Time: 45.32 seconds

...

================================================================================
EVALUATION SUMMARY
================================================================================
Total Questions: 10
Successful Responses: 9
Failed Responses: 1
Success Rate: 90.0%
Average Response Time: 42.15 seconds
================================================================================
```

## Question Format

The `johnson.jsonl` file contains questions in the following format:
```json
{
  "financebench_id": "financebench_id_00956",
  "company": "Johnson & Johnson",
  "question": "Are JnJ's FY2022 financials that of a high growth company?",
  "answer": "No, JnJ's FY2022 financials are not of a high growth company as sales grew by 1.3% in FY2022.",
  "evidence": [...],
  "doc_name": "JOHNSON_JOHNSON_2022_10K",
  "question_type": "domain-relevant"
}
```

## Evaluation Mode

The evaluation scripts automatically set `evaluation_mode: true` when making API requests. This parameter:
- Disables WebSocket updates during evaluation
- Prevents real-time dashboard interference
- Ensures consistent evaluation performance
- Reduces system overhead during testing

Example API request with evaluation mode:
```json
{
  "query": "Are JnJ's FY2022 financials that of a high growth company?",
  "search_query_type": "SEMANTIC",
  "graph_query_type": "local",
  "use_search": true,
  "use_graph": false,
  "use_web": true,
  "use_reasoning": false,
  "evaluation_mode": true
}
```

## Results Analysis

The evaluation results include:
- Response accuracy compared to expected answers
- Response time performance
- Error analysis for failed queries
- Success rate metrics

Use the generated `evaluation_results.json` file to analyze the agent team's performance on financial analysis tasks.

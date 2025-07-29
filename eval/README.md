# Johnson & Johnson Financial Analysis Evaluation

This directory contains evaluation scripts for testing the Agent Team's performance on Johnson & Johnson financial analysis questions.

## Files

- `johnson.jsonl` - Contains financial analysis questions from the FinanceBench dataset
- `evaluate_data_api.py` - Evaluation script that calls the `/query_team` API endpoint and runs response evaluation locally
- `evaluate_johnson_api.py` - Evaluation script that calls the `/query_team` API endpoint
- `evaluate_johnson.py` - Direct evaluation script that imports agent modules (may need setup)
- `test_setup.py` - Test script to verify the evaluation setup
- `test_single_question.py` - Quick test script to run a single question
- `requirements.txt` - Python dependencies for the evaluation scripts
- `evaluation_results.json` - Output file containing evaluation results (generated after running)

#### Output files generated after running `evaluate_data_api.py`
- `response_results.jsonl` - Agent team response for each qeury
- `evaluation_results.jsonl` - Agent team response along with the evaluation metrics
- `agent_converted_data.json` - Agent team conversation threads converted to evaluation-ready format

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your environment variables are set up in `../app/.env`:
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
python evaluate_data_api.py
```

### Method 2: Direct Module Evaluation

```bash
python evaluate_johnson.py
```

## Output

The evaluation script will:
1. Load questions from `johnson.jsonl`
2. Process each question through the agent team
3. Generate and save results to output files `response_results.jsonl`, `evaluation_results.jsonl`, `agent_converted_data.json`
4. Print a summary of the evaluation

## Sample Output

```
Starting Financial Analysis Evaluation
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

🪙 TOKEN USAGE SUMMARY:
Total Questions with Token Data: 9
Total Tokens Used: 22,635
Total Prompt Tokens: 20,793
Total Completion Tokens: 1,842
Average Tokens per Question: 7545.0
Average Prompt Tokens per Question: 6931.0
Average Completion Tokens per Question: 614.0
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

Questions in this format from other companies in the FinanceBench dataset are also suitable

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

The `evaluate_data_api.py` script performs comprehensive evaluation and generates three main output files:

### Output Files

1. **`response_results.jsonl`** - Raw agent responses for each query including:
   - Agent response text
   - Response time
   - Token usage statistics
   - Thread and run IDs
   - Context information

2. **`evaluation_results.jsonl`** - Detailed evaluation metrics for each response including:
   - All Azure AI evaluation scores (see metrics below)
   - Ground truth comparison
   - Question metadata (company, financebench_id)
   - Performance metrics (token usage, response times)

3. **`agent_converted_data.json`** - Agent conversation threads converted to evaluation-ready format

### Evaluation Metrics

The script uses Azure AI evaluation framework with the following evaluators:

#### Quality Metrics (with Reasoning Model Support)
- **IntentResolutionEvaluator** - Measures how well the response addresses the user's intent
- **TaskAdherenceEvaluator** - Evaluates adherence to the specific financial analysis task

#### Quality Metrics (Standard Model)
- **CoherenceEvaluator** - Assesses logical flow and consistency of the response
- **FluencyEvaluator** - Measures language quality and readability
- **RelevanceEvaluator** - Evaluates relevance to the financial question asked
- **GroundednessEvaluator** - Checks if response is grounded in provided context
- **SimilarityEvaluator** - Measures semantic similarity to expected answer

#### Accuracy Metrics
- **F1ScoreEvaluator** - Calculates F1 score against ground truth
- **MeteorScoreEvaluator** - Computes METEOR score for text similarity

### Performance Metrics

The evaluation also tracks:
- **Response Time** - Time taken to generate each response
- **Success Rate** - Percentage of successful API calls
- **Token Usage** - Detailed breakdown of prompt, completion, and total tokens
- **Error Analysis** - Categorization and tracking of failed queries

### Summary Statistics

The script provides comprehensive statistics including:
- Total questions processed
- Success/failure rates
- Average response times
- Token usage averages per question
- Cost implications based on token consumption

Use the generated `evaluation_results.jsonl` file to analyze individual question performance and the summary statistics to understand overall agent team effectiveness on financial analysis tasks.

## Generate Analysis Report

The `generate_report.py` script processes evaluation results JSONL files and generates comprehensive markdown reports with charts and statistical analysis.

### Usage

```bash
# Basic usage - generates report.md in current directory
python generate_report.py eval_data/*_evaluation_results.jsonl

# Specify output file and output directory
python generate_report.py eval_data/*_evaluation_results.jsonl --output my_report.md --output-dir outputs/

# Process specific files
python generate_report.py \
    eval_data/kg_drift_evaluation_results.jsonl \
    eval_data/kg_global_evaluation_results.jsonl \
    eval_data/rag_hybrid_evaluation_results.jsonl \
    --output analysis_report.md
```

### Required Dependencies

Make sure you have the following Python packages installed:
- pandas
- numpy
- matplotlib

### Output

The script generates:
1. A markdown report file (default: `evaluation_report.md`)
2. Agent evaluation summary CSV (`agent_evaluation_summary.csv`)
3. Response time chart (`response_time_chart.png`)
4. Overall score chart (`overall_score_chart.png`)

#### CSV File Structure

The `agent_evaluation_summary.csv` contains the following columns:
- **Agent_Type**: Name of the agent type
- **Average_Response_Time_Seconds**: Average response time in seconds
- **Overall_Average_Score**: Overall score averaged across all evaluation metrics
- **Intent_Resolution_Score**: Score for intent resolution (1-5 scale)
- **Task_Adherence_Score**: Score for task adherence (1-5 scale)
- **Coherence_Score**: Score for coherence (1-5 scale)
- **Fluency_Score**: Score for fluency (1-5 scale)
- **Relevance_Score**: Score for relevance (1-5 scale)
- **Groundedness_Score**: Score for groundedness (1-5 scale)
- **Similarity_Score**: Score for similarity (1-5 scale)
- **F1_Score**: F1 score (0-1 scale)
- **Meteor_Score**: Meteor score (0-1 scale)
- **Total_Records**: Number of evaluation records for the agent

#### Report Sections

The generated report includes:

##### Section 1: Average Response Time Analysis
- Response time summary table
- Response time bar chart
- Performance categories (High-speed, Medium-speed, Slower tier)
- Key insights

##### Section 2: Overall Evaluation Metrics Analysis
- Overall score summary table with rankings
- Overall score bar chart
- Evaluation metrics breakdown
- Quality vs Speed trade-off analysis
- Individual metric performance table
- Top performers by metric
"""
Evaluation script for financial analysis using Agent Team API.

This script reads questions from a jsonl file and evaluates the agent team's 
performance on financial analysis tasks by calling the /query_team API endpoint.
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
env_file_path = Path(__file__).parent.parent / "app" / ".env"
if env_file_path.exists():
    load_dotenv(env_file_path)

# Configuration
JSONL_FILE_PATH = Path(__file__).parent / "johnson.jsonl"  # Path to the input JSONL file with questions
RESULTS_FILE_PATH = Path(__file__).parent / "response_results.jsonl"  #Path to save response results
AGENT_RESULTS_FILE_PATH = Path(__file__).parent / "agent_converted_data.json"  
EVAL_RESULTS_FILE_PATH = Path(__file__).parent / "evaluation_results.jsonl"  # Path to save evaluation results

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")

# default payload for API requests
DEFAULT_PAYLOAD = {
        "search_query_type": "SIMPLE",
        "graph_query_type": "drift",
        "use_search": True,
        "use_graph": False,
        "use_web": False,
        "use_reasoning": False,
        "evaluation_mode": True
    }

def load_questions_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data = json.loads(line.strip())
                    questions.append(data)
        print(f"Loaded {len(questions)} questions from {file_path}")
        return questions
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []
    
def call_query_team_api(question: str) -> Dict[str, Any]:
    """Call the /query_team API endpoint."""
    url = f"{API_BASE_URL}/query_team"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if available
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    # Prepare the payload
    payload = {
        "query": question,
        "search_query_type": DEFAULT_PAYLOAD["search_query_type"],
        "graph_query_type": DEFAULT_PAYLOAD["graph_query_type"],
        "use_search": DEFAULT_PAYLOAD["use_search"],
        "use_graph": DEFAULT_PAYLOAD["use_graph"],
        "use_web": DEFAULT_PAYLOAD["use_web"],
        "use_reasoning": DEFAULT_PAYLOAD["use_reasoning"],
        "evaluation_mode": DEFAULT_PAYLOAD["evaluation_mode"]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=18000)  # 5 hour timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def evaluate_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question using the agent team API."""
    question = question_data["question"]
    expected_answer = question_data["answer"]
    financebench_id = question_data["financebench_id"]
    company = question_data["company"]
    
    print("\n" + "="*80)
    print(f"Evaluating Question ID: {financebench_id}")
    print(f"Company: {company}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Call the API
        api_response = call_query_team_api(question)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if "error" in api_response:
            print(f"\nAPI Error: {api_response['error']}")
            result = {
                "financebench_id": financebench_id,
                "company": company,
                "question": question,
                "expected_answer": expected_answer,
                "agent_response": f"API Error: {api_response['error']}",
                "response_time": response_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error",
                "error": api_response['error'],
                "token_usage": None
            }
        else:
            agent_response = api_response.get("response", "No response received")
            token_usage = api_response.get("token_usage", {})
            print(f"\nAgent Response:\n{agent_response}")
            print(f"\nResponse Time: {response_time:.2f} seconds")
            
            result = {
                "financebench_id": financebench_id,
                "company": company,
                "query": question,
                "ground_truth": expected_answer,
                "context": api_response.get("context"),
                "response": agent_response,
                "thread_id": api_response.get("thread_id"),
                "run_id": api_response.get("run_id"),
                "response_time": response_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "token_usage": token_usage
            }
        
        return result
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"\nError processing question: {str(e)}")
        
        result = {
            "financebench_id": financebench_id,
            "company": company,
            "query": question,
            "ground_truth": expected_answer,
            "response": f"Error: {str(e)}",
            "response_time": response_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "error": str(e),
            "token_usage": None
        }
        
        return result
    
def save_results(results: List[Dict[str, Any]], file_path: Path):
    """Save evaluation results to JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for result in results:
                json.dump(result, file, ensure_ascii=False)
                file.write('\n')
        print(f"\nResults saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def save_agent_converted_data(converted_data: Dict[str, Any], file_path: Path):
    """Save converted agent data to JSON file, appending to existing data."""
    try:
        # Read existing data if file exists
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    existing_data = json.load(file)
                    # Ensure it's a list
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # Append new data
        existing_data.append(converted_data)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=2)
        print(f"\nAgent converted data appended to {file_path}")
    except Exception as e:
        print(f"Error saving agent converted data: {e}")

def save_evaluation_results(eval_result: Dict[str, Any], file_path: Path):
    """Save evaluation results to JSONL file, appending each result as a new line."""
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            json.dump(eval_result, file, ensure_ascii=False)
            file.write('\n')
        print(f"\nEvaluation result appended to {file_path}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")

def print_summary(results: List[Dict[str, Any]]):
    """Print evaluation summary."""
    total_questions = len(results)
    successful_questions = len([r for r in results if r["status"] == "success"])
    failed_questions = total_questions - successful_questions
    
    if successful_questions > 0:
        avg_response_time = sum(r["response_time"] for r in results if r["status"] == "success") / successful_questions
    else:
        avg_response_time = 0

    # Calculate token usage statistics
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    questions_with_tokens = 0

    for result in results:
        if result["status"] == "success" and result.get("token_usage"):
            token_usage = result["token_usage"]
            total_tokens += token_usage.get("total_tokens", 0)
            total_prompt_tokens += token_usage.get("total_prompt_tokens", 0)
            total_completion_tokens += token_usage.get("total_completion_tokens", 0)
            questions_with_tokens += 1
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {total_questions}")
    print(f"Successful Responses: {successful_questions}")
    print(f"Failed Responses: {failed_questions}")
    print(f"Success Rate: {(successful_questions/total_questions)*100:.1f}%")
    print(f"Average Response Time: {avg_response_time:.2f} seconds")
    print("\n🪙 TOKEN USAGE SUMMARY:")
    print(f"Total Questions with Token Data: {questions_with_tokens}")
    print(f"Total Tokens Used: {total_tokens:,}")
    print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"Total Completion Tokens: {total_completion_tokens:,}")
    if questions_with_tokens > 0:
        print(f"Average Tokens per Question: {total_tokens/questions_with_tokens:.1f}")
        print(f"Average Prompt Tokens per Question: {total_prompt_tokens/questions_with_tokens:.1f}")
        print(f"Average Completion Tokens per Question: {total_completion_tokens/questions_with_tokens:.1f}")
    print("="*80)

def test_api_connection():
    """Test API connection."""
    try:
        url = f"{API_BASE_URL}/health"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print(f"✅ API connection successful: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False
    
def main():
    """Main evaluation function."""
    print("Starting Financial Analysis Evaluation")
    print("="*80)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    
    # Test API connection
    if not test_api_connection():
        print("Cannot connect to API. Please ensure the server is running.")
        sys.exit(1)
    
    # Load questions from JSONL file
    questions = load_questions_from_jsonl(JSONL_FILE_PATH)
    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)
    
    # Evaluate each question
    results = []
    for i, question_data in enumerate(questions, 1):
        print(f"\nProcessing question {i}/{len(questions)}")
        result = evaluate_question(question_data)
        results.append(result)
        print(f"Finished processing question {i}.")
        # Add a small delay between questions to avoid rate limiting
        if i < len(questions):
            time.sleep(10)  # 10 second delay between requests
        if i == 3:   # For testing, limit to first 3 questions
            break
        # break    #remove this break to evaluate all questions
    # Save results
    save_results(results, RESULTS_FILE_PATH)
    
    # Print summary
    print_summary(results)
    
    print("\nResponses to questions completed!")

    # Azure AI Projects evaluation (optional)
    try:
        import os
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient

        # Check if Azure environment variables are available
        required_env_vars = ["PROJECT_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "MODEL_DEPLOYMENT_NAME"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"\n⚠️  Skipping Azure AI Projects evaluation. Missing environment variables: {missing_vars}")
            return

        # Required environment variables:
        endpoint = os.environ["PROJECT_ENDPOINT"] # https://<account>.services.ai.azure.com/api/projects/<project>
        model_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] # https://<account>.services.ai.azure.com
        model_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"] # E.g. gpt-4o-mini

        model_config = {
            "azure_deployment": os.getenv("MODEL_DEPLOYMENT_NAME"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),

        }

        reasoning_model_config = {
            "azure_deployment": os.getenv("REASONING_MODEL_DEPLOYMENT_NAME"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": "2024-12-01-preview",  # Ensure using the correct API version

        }

        # Create the project client (Foundry project and credentials):
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )

        # Import evaluator models
        from azure.ai.evaluation import AIAgentConverter, IntentResolutionEvaluator, TaskAdherenceEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, GroundednessEvaluator, SimilarityEvaluator, F1ScoreEvaluator, MeteorScoreEvaluator

        # Initialize the converter for Azure AI agents.
        converter = AIAgentConverter(project_client)

        # Evaluators with reasoning model support
        quality_evaluators = {evaluator.__name__: evaluator(model_config=reasoning_model_config, is_reasoning_model=True) for evaluator in [IntentResolutionEvaluator, TaskAdherenceEvaluator]}

        # Other evaluators do not support reasoning models
        quality_evaluators.update({evaluator.__name__: evaluator(model_config=model_config) for evaluator in [CoherenceEvaluator, FluencyEvaluator, RelevanceEvaluator, GroundednessEvaluator, SimilarityEvaluator]})

        # Add F1 and Meteor evaluators
        quality_evaluators.update({evaluator.__name__: evaluator() for evaluator in [F1ScoreEvaluator, MeteorScoreEvaluator]})

        for i, result in enumerate(results):
            if result["status"] != "success":
                continue  # Skip failed results
            print(f"\nEvaluating response {i + 1}/{len(results)}")
            thread_id = result['thread_id']
            run_id = result['run_id']

            converted_data = converter.convert(thread_id, run_id)
            question_id = result['financebench_id']

            print(f"Evaluating question ID: {question_id}")

            # Save converted data to JSON file
            save_agent_converted_data(converted_data, AGENT_RESULTS_FILE_PATH)

            all_eval_results = []
            for name, evaluator in quality_evaluators.items():
                print(f"Evaluating with {name}...")
                if name in ["GroundednessEvaluator", "SimilarityEvaluator", "F1ScoreEvaluator", "MeteorScoreEvaluator"]:
                    # Special handling for evaluators that require specific parameters
                    eval_result = evaluator(**result)
                else:
                    eval_result = evaluator(**converted_data)
                print(json.dumps(eval_result, indent=4))
                
                # Add eval name to evaluation result
                eval_result['evaluator_name'] = name
                all_eval_results.append(eval_result)
            
            # Combine all evaluation results into a single dictionary
            final_eval_result = {
                "question_id": question_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "response_time": result['response_time'],
                "timestamp": result['timestamp'],
                "status": result['status'],
                "evaluations": all_eval_results,
                "ground_truth": result['ground_truth'],
                "response": result['response'],
                "context": result['context'] if 'context' in result else None,
                "company": result['company'],
                "query": result['query'] if 'query' in result else None,
                "token_usage": result.get('token_usage', None)
            }
            # Save evaluation results to JSONL file
            save_evaluation_results(final_eval_result, EVAL_RESULTS_FILE_PATH)
            print(f"✅Saved evaluation result for question ID {question_id}.")

    except ImportError:
        print("\n⚠️  Azure AI Projects libraries not available. Skipping evaluation.")
        return
    except Exception as e:
        print(f"\n❌ Error running Azure AI evaluation: {e}")
        return

if __name__ == "__main__":
    main()
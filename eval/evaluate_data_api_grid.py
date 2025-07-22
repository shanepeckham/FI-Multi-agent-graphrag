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
JSONL_FILE_PATH = Path(__file__).parent / "pepsico.jsonl"  # Path to the input JSONL file with questions


# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")

# default payload for API requests
DEFAULT_PAYLOAD = {
        "search_query_type": "SEMANTIC",
        "graph_query_type": "local",
        "use_search": True,
        "use_graph": False,
        "use_web": False,
        "use_reasoning": False,
        "evaluation_mode": True
    }

def generate_config_combinations() -> List[Dict[str, Any]]:
    """Generate all combinations of agent options, search query types, and graph query types."""
    agent_options = [
        {'use_search': True, 'use_graph': False, 'use_web': False, 'use_reasoning': False},
        {'use_search': False, 'use_graph': True, 'use_web': False, 'use_reasoning': False},
    ]
    search_query_type_options = ['SIMPLE', 'SEMANTIC']
    graph_query_type_options = ['local', 'drift', 'global']
    
    configurations = []
    
    for agent_option in agent_options:
        for search_query_type in search_query_type_options:
            for graph_query_type in graph_query_type_options:
                config = {
                    **agent_option,
                    "search_query_type": search_query_type,
                    "graph_query_type": graph_query_type
                }
                configurations.append(config)
    
    return configurations

def get_config_name(config: Dict[str, Any]) -> str:
    """Generate a descriptive name for the configuration."""
    agent_type = "search" if config['use_search'] else "graph" if config['use_graph'] else "unknown"
    search_type = config['search_query_type'].lower()
    graph_type = config['graph_query_type']
    
    return f"{agent_type}_{search_type}_{graph_type}"

def get_config_file_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Path]:
    """Generate file paths for a specific configuration."""
    config_name = get_config_name(config)
    
    # Create grid_search directory within the base directory
    grid_search_dir = base_dir / "grid_search"
    grid_search_dir.mkdir(exist_ok=True)
    
    return {
        "results": grid_search_dir / f"response_results_{config_name}.jsonl",
        "agent_converted": grid_search_dir / f"agent_converted_data_{config_name}.json",
        "eval_results": grid_search_dir / f"evaluation_results_{config_name}.jsonl"
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
    
def call_query_team_api(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Call the /query_team API endpoint with specific configuration."""
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
        "search_query_type": config["search_query_type"],
        "graph_query_type": config["graph_query_type"],
        "use_search": config["use_search"],
        "use_graph": config["use_graph"],
        "use_web": config["use_web"],
        "use_reasoning": config["use_reasoning"],
        "evaluation_mode": True
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=18000)  # 5 hour timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
def evaluate_question(question_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question using the agent team API with specific configuration."""
    question = question_data["question"]
    expected_answer = question_data["answer"]
    financebench_id = question_data["financebench_id"]
    company = question_data["company"]
    
    print("\n" + "="*80)
    print(f"Evaluating Question ID: {financebench_id}")
    print(f"Company: {company}")
    print(f"Configuration: {config}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Call the API
        api_response = call_query_team_api(question, config)
        
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
                "configuration": config
            }
        else:
            agent_response = api_response.get("response", "No response received")
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
                "configuration": config
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
            "configuration": config
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
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {total_questions}")
    print(f"Successful Responses: {successful_questions}")
    print(f"Failed Responses: {failed_questions}")
    print(f"Success Rate: {(successful_questions/total_questions)*100:.1f}%")
    print(f"Average Response Time: {avg_response_time:.2f} seconds")
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
    
def run_azure_evaluation(results: List[Dict[str, Any]], file_paths: Dict[str, Path]):
    """Run Azure AI Projects evaluation on the results."""
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
            save_agent_converted_data(converted_data, file_paths["agent_converted"])

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
                "evaluations": all_eval_results,
                "ground_truth": result['ground_truth'],
                "response": result['response'],
                "context": result['context'] if 'context' in result else None,
                "company": result['company'],
                "query": result['query'] if 'query' in result else None,
            }

            # Save evaluation results to JSONL file
            save_evaluation_results(final_eval_result, file_paths["eval_results"])
            print(f"✅Saved evaluation result for question ID {question_id}.")

    except ImportError:
        print("\n⚠️  Azure AI Projects libraries not available. Skipping evaluation.")
        return
    except Exception as e:
        print(f"\n❌ Error setting up Azure AI Projects: {e}")
        return

def run_evaluation_for_config(questions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run evaluation for a specific configuration."""
    config_name = get_config_name(config)
    base_dir = Path(__file__).parent
    file_paths = get_config_file_paths(config, base_dir)
    
    print(f"\n{'='*100}")
    print(f"RUNNING EVALUATION FOR CONFIGURATION: {config_name}")
    print(f"Configuration: {config}")
    print(f"{'='*100}")
    
    # Evaluate each question
    results = []
    for i, question_data in enumerate(questions, 1):
        print(f"\nProcessing question {i}/{len(questions)} for config {config_name}")
        result = evaluate_question(question_data, config)
        results.append(result)
        
        # Add a small delay between questions to avoid rate limiting
        if i < len(questions):
            time.sleep(5)  # 5 second delay between requests
        break    # Uncomment this break to evaluate only one question per config for testing
    
    # Save results
    save_results(results, file_paths["results"])
    
    # Print summary for this configuration
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    print_summary(results)
    
    # Run Azure evaluation if available
    run_azure_evaluation(results, file_paths)
    
    print(f"\n✅ Completed evaluation for configuration: {config_name}")
    
    return results

def main():
    """Main evaluation function with grid search."""
    print("Starting Financial Analysis Grid Evaluation")
    print("="*80)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key configured: {'Yes' if API_KEY else 'No'}")
    
    # Create grid_search directory
    script_dir = Path(__file__).parent
    grid_search_dir = script_dir / "grid_search"
    grid_search_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {grid_search_dir}")
    
    # Test API connection
    if not test_api_connection():
        print("Cannot connect to API. Please ensure the server is running.")
        sys.exit(1)
    
    # Load questions from JSONL file
    questions = load_questions_from_jsonl(JSONL_FILE_PATH)
    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)
    
    # Generate all configuration combinations
    configurations = generate_config_combinations()
    
    print(f"\n📊 Generated {len(configurations)} configuration combinations:")
    for i, config in enumerate(configurations, 1):
        config_name = get_config_name(config)
        print(f"  {i}. {config_name}: {config}")
    
    print(f"\n🚀 Starting grid evaluation with {len(questions)} questions across {len(configurations)} configurations...")
    
    all_results = []
    for i, config in enumerate(configurations, 1):
        print(f"\n🔄 Running configuration {i}/{len(configurations)}")
        config_results = run_evaluation_for_config(questions, config)
        all_results.extend(config_results)
        
        # Add a longer delay between configurations to avoid overwhelming the API
        if i < len(configurations):
            print(f"\n⏱️  Waiting 15 seconds before next configuration...")
            time.sleep(15)
    
    # Print overall summary
    print(f"\n{'='*100}")
    print("OVERALL GRID EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"Total configurations evaluated: {len(configurations)}")
    print(f"Total questions per configuration: {len(questions)}")
    print(f"Total evaluations performed: {len(all_results)}")
    print(f"Results saved to: {grid_search_dir}")
    
    # Summary by configuration
    for config in configurations:
        config_name = get_config_name(config)
        config_results = [r for r in all_results if r.get("configuration") == config]
        successful = len([r for r in config_results if r["status"] == "success"])
        total = len(config_results)
        success_rate = (successful/total)*100 if total > 0 else 0
        print(f"  {config_name}: {successful}/{total} successful ({success_rate:.1f}%)")
    
    print(f"\n✅ Grid evaluation completed! Check the grid_search folder ({grid_search_dir}) for detailed results.")

if __name__ == "__main__":
    main()


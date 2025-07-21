#!/usr/bin/env python3
"""
Evaluation script for Johnson & Johnson financial analysis using Agent Team API.

This script reads questions from johnson.jsonl file and evaluates the agent team's 
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
JSONL_FILE_PATH = Path(__file__).parent / "johnson.jsonl"
RESULTS_FILE_PATH = Path(__file__).parent / "evaluation_results.jsonl"
AGENT_RESULTS_FILE_PATH = Path(__file__).parent / "agent_converted_data.json"

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY")

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
    
    payload = {
        "query": question,
        "search_query_type": "SEMANTIC",
        "graph_query_type": "local",
        "use_search": True,
        "use_graph": False,
        "use_web": False,
        "use_reasoning": False,
        "evaluation_mode": True
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
                "error": api_response['error']
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
                "status": "success"
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
            "error": str(e)
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
    """Save converted agent data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(converted_data, file, ensure_ascii=False, indent=2)
        print(f"\nAgent converted data saved to {file_path}")
    except Exception as e:
        print(f"Error saving agent converted data: {e}")

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

def main():
    """Main evaluation function."""
    print("Starting Johnson & Johnson Financial Analysis Evaluation")
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
        
        # Add a small delay between questions to avoid rate limiting
        if i < len(questions):
            time.sleep(3)  # 3 second delay between requests
        break
    # Save results
    save_results(results, RESULTS_FILE_PATH)


    
    # Print summary
    print_summary(results)
    
    print("\nEvaluation completed!")
    
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
            "api_version": "2025-01-01-preview",  # Ensure using the correct API version

        }


        # Dataset configuration
        dataset_name = results[0]["financebench_id"] 
        dataset_version = "1.13"

        print(f"\n🔄 Starting Azure AI Projects evaluation...")
        print(f"Dataset: {dataset_name} v{dataset_version}")

        # Create the project client (Foundry project and credentials):
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )

        # Try to delete existing dataset (ignore errors if it doesn't exist)
        try:
            project_client.datasets.delete(
                name=dataset_name,
                version=dataset_version,
            )
            print(f"✅ Deleted existing dataset: {dataset_name}")
        except Exception as e:
            print(f"ℹ️  No existing dataset to delete: {e}")

        # Upload the JSONL file
        print(f"📤 Uploading dataset from: {RESULTS_FILE_PATH}")
        data_id = project_client.datasets.upload_file(
            file_path=str(RESULTS_FILE_PATH),
            name=dataset_name,
            version=dataset_version,
        ).id

        print(f"✅ Dataset uploaded with ID: {data_id}")

        # Import evaluator models
        from azure.ai.projects.models import (
            EvaluatorConfiguration,
            EvaluatorIds,
            Evaluation,
            InputDataset
        )

        from azure.ai.evaluation import AIAgentConverter, IntentResolutionEvaluator, TaskAdherenceEvaluator, ToolCallAccuracyEvaluator, RelevanceEvaluator, CoherenceEvaluator, CodeVulnerabilityEvaluator, ContentSafetyEvaluator, IndirectAttackEvaluator, FluencyEvaluator

        # Initialize the converter for Azure AI agents.
        converter = AIAgentConverter(project_client)

        thread_id = results[0]["thread_id"]
        run_id = results[0]["run_id"]

        converted_data = converter.convert(thread_id, run_id)

        # Save converted data to JSON file
        save_agent_converted_data(converted_data, AGENT_RESULTS_FILE_PATH)

        # Dataset configuration
        agent_dataset_name = "agent_"+ results[0]["financebench_id"] 


        print(f"📤 Uploading agent dataset from: {AGENT_RESULTS_FILE_PATH}")
        agent_data_id = project_client.datasets.upload_file(
            file_path=str(AGENT_RESULTS_FILE_PATH),
            name=agent_dataset_name,
            version=dataset_version,
        ).id

        # Evaluators with reasoning model support
        quality_evaluators = {evaluator.__name__: evaluator(model_config=reasoning_model_config, is_reasoning_model=True) for evaluator in [IntentResolutionEvaluator, TaskAdherenceEvaluator, ToolCallAccuracyEvaluator]}

        # Other evaluators do not support reasoning models 
        quality_evaluators.update({ evaluator.__name__: evaluator(model_config=model_config) for evaluator in [CoherenceEvaluator, FluencyEvaluator, RelevanceEvaluator]})

 
        #safety_evaluators = {evaluator.__name__: evaluator(azure_ai_project=project_client, credential=DefaultAzureCredential()) for evaluator in[ContentSafetyEvaluator, IndirectAttackEvaluator, CodeVulnerabilityEvaluator]}

        # Reference the quality and safety evaluator list above.
        #quality_and_safety_evaluators = {**quality_evaluators, **safety_evaluators}

        for name, evaluator in quality_evaluators.items():
            result = evaluator(**converted_data)
            print(name)
            print(json.dumps(result, indent=4)) 

        # Built-in evaluator configurations:
        evaluators = {
            "relevance": EvaluatorConfiguration(
                id=EvaluatorIds.RELEVANCE.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "ground_truth": "${data.ground_truth}",
                },
            ),
            "groundedness": EvaluatorConfiguration(
                id=EvaluatorIds.GROUNDEDNESS.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "ground_truth": "${data.ground_truth}",
                    "context": "${data.context}",
                },
            ),

        }

        # Create an evaluation with the dataset and evaluators specified.
        evaluation = Evaluation(
            display_name=dataset_name,
            description="Finbench Johnson evaluation",
            data=InputDataset(id=data_id),
            evaluators=evaluators,
            tags="finbench, johnson, RAG-Agent",
        )



        # Run the evaluation.
        print("🚀 Starting evaluation...")
        evaluation_response = project_client.evaluations.create(
            evaluation,
            headers={
                "model-endpoint": model_endpoint,
                "api-key": model_api_key,
            },
        )

        # Create an evaluation with the dataset and evaluators specified.
        agent_evaluation = Evaluation(
            display_name=agent_dataset_name,
            description="Agent Finbench Johnson evaluation",
            data=InputDataset(id=agent_data_id),
            evaluators=quality_evaluators,
            tags="finbench, johnson, RAG-Agent",
        )

        # Run the evaluation.
        print("🚀 Starting agent evaluation...")
        agent_evaluation_response = project_client.evaluations.create(
            agent_evaluation,
            headers={
                "model-endpoint": model_endpoint,
                "api-key": model_api_key,
                "model-deployment-name": os.getenv("REASONING_MODEL_DEPLOYMENT_NAME")
            },
        )

        print(f"✅ Created evaluation: {evaluation_response.name}{agent_evaluation_response.name}")
        print(f"📊 Status: {evaluation_response.status} {agent_evaluation_response.status}")
        print(f"🔗 Evaluation ID: {evaluation_response.id}")

    except ImportError:
        print("\n⚠️  Azure AI Projects libraries not available. Skipping evaluation.")
        return
    except Exception as e:
        print(f"\n❌ Error setting up Azure AI Projects: {e}")
        return


if __name__ == "__main__":
    main()

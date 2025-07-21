#!/usr/bin/env python3
"""
Evaluation script for Johnson & Johnson financial analysis using Agent Team.

This script reads questions from johnson.jsonl file and evaluates the agent team's 
performance on financial analysis tasks.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

# Import required modules from the app
from agent_team import AgentTeam
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ToolSet
import yaml

# Load environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
env_file_path = Path(__file__).parent.parent / "apple" / ".env"
if env_file_path.exists():
    load_dotenv(env_file_path)

# Configuration
JSONL_FILE_PATH = Path(__file__).parent / "johnson.jsonl"
RESULTS_FILE_PATH = Path(__file__).parent / "evaluation_results.json"
CONFIG_FILE_PATH = Path(__file__).parent.parent / "app" / "agent_team_config.yaml"

# Azure configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT", "https://fiagent.cognitiveservices.azure.com/")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")

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

def setup_agent_team(agents_client: AgentsClient) -> AgentTeam:
    """Set up the agent team for evaluation."""
    # Create agent team
    agent_team = AgentTeam("evaluation_team", agents_client=agents_client)
    
    # Load configuration
    with open(CONFIG_FILE_PATH, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # Create basic toolset (simplified for evaluation)
    toolset = ToolSet()
    
    # Add RAG agent (simplified for evaluation)
    rag_instructions = config["RAG_AGENT_INSTRUCTIONS"]
    agent_team.add_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="RAG-agent-multi",
        instructions=rag_instructions,
        toolset=toolset,
        can_delegate=True
    )
    
    # Add KG agent (simplified for evaluation)
    kg_instructions = config["KG_AGENT_INSTRUCTIONS"]
    agent_team.add_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="KG-agent-multi",
        instructions=kg_instructions,
        toolset=toolset,
        can_delegate=True
    )
    
    # Add Bing agent (simplified for evaluation)
    bing_instructions = config["BING_AGENT_INSTRUCTIONS"]
    agent_team.add_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="Bing-agent-multi",
        instructions=bing_instructions,
        toolset=toolset,
        can_delegate=True
    )
    
    # Assemble the team
    agent_team.assemble_team()
    
    return agent_team

def evaluate_question(agent_team: AgentTeam, question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question using the agent team."""
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
        # Process the request using the agent team
        agent_response = agent_team.process_request(question)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"\nAgent Response:\n{agent_response}")
        print(f"\nResponse Time: {response_time:.2f} seconds")
        
        # Create evaluation result
        result = {
            "financebench_id": financebench_id,
            "company": company,
            "question": question,
            "expected_answer": expected_answer,
            "agent_response": agent_response,
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
            "question": question,
            "expected_answer": expected_answer,
            "agent_response": f"Error: {str(e)}",
            "response_time": response_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "error": str(e)
        }
        
        return result

def save_results(results: List[Dict[str, Any]], file_path: Path):
    """Save evaluation results to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

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

def main():
    """Main evaluation function."""
    print("Starting Johnson & Johnson Financial Analysis Evaluation")
    print("="*80)
    
    # Check required environment variables
    if not AZURE_OPENAI_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load questions from JSONL file
    questions = load_questions_from_jsonl(JSONL_FILE_PATH)
    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)
    
    # Initialize Azure agents client
    try:
        agents_client = AgentsClient(
            endpoint=PROJECT_ENDPOINT,
            credential=AZURE_OPENAI_API_KEY
        )
        print(f"Initialized Azure Agents Client with endpoint: {PROJECT_ENDPOINT}")
    except Exception as e:
        print(f"Error initializing Azure Agents Client: {e}")
        sys.exit(1)
    
    # Setup agent team
    try:
        agent_team = setup_agent_team(agents_client)
        print("Agent team assembled successfully")
    except Exception as e:
        print(f"Error setting up agent team: {e}")
        sys.exit(1)
    
    # Evaluate each question
    results = []
    for i, question_data in enumerate(questions, 1):
        print(f"\nProcessing question {i}/{len(questions)}")
        result = evaluate_question(agent_team, question_data)
        results.append(result)
        
        # Add a small delay between questions to avoid rate limiting
        if i < len(questions):
            time.sleep(2)
    
    # Save results
    save_results(results, RESULTS_FILE_PATH)
    
    # Print summary
    print_summary(results)
    
    # Cleanup
    try:
        agent_team.dismantle_team()
        print("\nAgent team dismantled successfully")
    except Exception as e:
        print(f"Error dismantling agent team: {e}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()

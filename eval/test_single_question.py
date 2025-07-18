#!/usr/bin/env python3
"""
Quick test script to run a single question from the johnson.jsonl file.
"""

import json
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_file_path = Path(__file__).parent.parent / "apple" / ".env"
if env_file_path.exists():
    load_dotenv(env_file_path)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

def load_sample_question():
    """Load the first question from johnson.jsonl."""
    jsonl_file = Path(__file__).parent / "johnson.jsonl"
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        line = file.readline().strip()
        if line:
            return json.loads(line)
    
    return None

def test_single_question():
    """Test a single question with the agent team."""
    question_data = load_sample_question()
    if not question_data:
        print("❌ Could not load sample question")
        return
    
    question = question_data["question"]
    expected_answer = question_data["answer"]
    
    print("Testing single question...")
    print("="*60)
    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")
    print("="*60)
    
    # Test API connection
    try:
        health_url = f"{API_BASE_URL}/health"
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        print("✅ API is healthy")
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return
    
    # Make the request
    url = f"{API_BASE_URL}/query_team"
    headers = {"Content-Type": "application/json"}
    
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    payload = {
        "query": question,
        "search_query_type": "SIMPLE",
        "graph_query_type": "local",
        "use_search": True,
        "use_graph": True,
        "use_web": True,
        "use_reasoning": False
    }
    
    print("\n🤖 Querying agent team...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        
        end_time = time.time()
        response_time = end_time - start_time
        
        result = response.json()
        agent_response = result.get("response", "No response")
        
        print(f"\n✅ Response received in {response_time:.2f} seconds")
        print("\nAgent Response:")
        print("-" * 60)
        print(agent_response)
        print("-" * 60)
        
        # Basic comparison
        if expected_answer.lower() in agent_response.lower():
            print("\n✅ Response contains expected answer elements")
        else:
            print("\n⚠️  Response may not match expected answer")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON decode error: {e}")

def main():
    """Main function."""
    print("Single Question Test for Agent Team")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_single_question.py")
        print("Tests a single question from johnson.jsonl against the agent team API.")
        print("\nMake sure the API server is running first:")
        print("cd ../app && python main.py")
        return
    
    test_single_question()

if __name__ == "__main__":
    main()

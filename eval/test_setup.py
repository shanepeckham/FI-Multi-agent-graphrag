#!/usr/bin/env python3
"""
Test script to verify the evaluation setup is working correctly.
"""

import json
import sys
from pathlib import Path

def test_jsonl_loading():
    """Test if we can load the JSONL file."""
    jsonl_file = Path(__file__).parent / "johnson.jsonl"
    
    if not jsonl_file.exists():
        print(f"❌ JSONL file not found: {jsonl_file}")
        return False
    
    try:
        questions = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line.strip())
                    questions.append(data)
        
        print(f"✅ Successfully loaded {len(questions)} questions from JSONL file")
        
        # Print first question as example
        if questions:
            first_question = questions[0]
            print(f"\nSample question:")
            print(f"ID: {first_question.get('financebench_id', 'N/A')}")
            print(f"Company: {first_question.get('company', 'N/A')}")
            print(f"Question: {first_question.get('question', 'N/A')[:100]}...")
            print(f"Answer: {first_question.get('answer', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading JSONL file: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available."""
    try:
        import requests
        print("✅ requests module available")
    except ImportError:
        print("❌ requests module not available. Run: pip install requests")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv module available")
    except ImportError:
        print("❌ python-dotenv module not available. Run: pip install python-dotenv")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing evaluation setup...")
    print("="*50)
    
    success = True
    
    # Test dependencies
    if not test_dependencies():
        success = False
    
    print()
    
    # Test JSONL loading
    if not test_jsonl_loading():
        success = False
    
    print()
    
    if success:
        print("✅ All tests passed! The evaluation setup is ready.")
        print("\nNext steps:")
        print("1. Start the GraphRAG API server: cd ../app && python main.py")
        print("2. Run the evaluation: python evaluate_johnson_api.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

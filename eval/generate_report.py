#!/usr/bin/env python3
"""
Generate evaluation analysis markdown report from JSONL files

This script processes agent evaluation results from JSONL files and generates
a comprehensive markdown report with sections on response time analysis and evaluation
metrics analysis.

Usage:
    python3 generate_report.py file1.jsonl file2.jsonl ... [options]

Example:
    python3 generate_report.py eval_data/kg_drift_evaluation_results.jsonl eval_data/rag_semantic_evaluation_results.jsonl --output report.md --output-dir charts/

Requirements:
    - pandas
    - numpy
    - matplotlib

The script expects JSONL files with the following structure:
    - Each line contains a JSON object with evaluation results
    - Required fields: response_time, evaluations, status
    - Evaluations should contain metric scores from various evaluators
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Any
import os


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def extract_metrics_from_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract evaluation metrics from the evaluations list"""
    metrics = {}
    
    for eval_item in evaluations:
        evaluator_name = eval_item.get('evaluator_name', '')
        
        if evaluator_name == 'IntentResolutionEvaluator':
            metrics['intent_resolution'] = eval_item.get('intent_resolution', 0.0)
        elif evaluator_name == 'TaskAdherenceEvaluator':
            metrics['task_adherence'] = eval_item.get('task_adherence', 0.0)
        elif evaluator_name == 'CoherenceEvaluator':
            metrics['coherence'] = eval_item.get('coherence', 0.0)
        elif evaluator_name == 'FluencyEvaluator':
            metrics['fluency'] = eval_item.get('fluency', 0.0)
        elif evaluator_name == 'RelevanceEvaluator':
            metrics['relevance'] = eval_item.get('relevance', 0.0)
        elif evaluator_name == 'GroundednessEvaluator':
            metrics['groundedness'] = eval_item.get('groundedness', 0.0)
        elif evaluator_name == 'SimilarityEvaluator':
            metrics['similarity'] = eval_item.get('similarity', 0.0)
        elif evaluator_name == 'F1ScoreEvaluator':
            metrics['f1_score'] = eval_item.get('f1_score', 0.0)
        elif evaluator_name == 'MeteorScoreEvaluator':
            metrics['meteor_score'] = eval_item.get('meteor_score', 0.0)
    
    return metrics


def process_agent_data(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process all agent evaluation files and calculate statistics"""
    agent_stats = {}
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue
            
        # Extract agent type from filename
        filename = os.path.basename(file_path)
        if '_evaluation_results.jsonl' in filename:
            agent_type = filename.replace('_evaluation_results.jsonl', '')
        else:
            print(f"Warning: Unexpected filename format: {filename}")
            continue
        
        data = load_jsonl_file(file_path)
        if not data:
            print(f"Warning: No data found in {file_path}")
            continue
        
        # Extract metrics for each record
        response_times = []
        all_metrics = {
            'intent_resolution': [],
            'task_adherence': [],
            'coherence': [],
            'fluency': [],
            'relevance': [],
            'groundedness': [],
            'similarity': [],
            'f1_score': [],
            'meteor_score': []
        }
        
        for record in data:
            if record.get('status') == 'success':
                response_times.append(record.get('response_time', 0))
                
                metrics = extract_metrics_from_evaluations(record.get('evaluations', []))
                for metric, value in metrics.items():
                    if metric in all_metrics:
                        all_metrics[metric].append(value)
        
        # Calculate averages
        avg_response_time = np.mean(response_times) if response_times else 0
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[metric] = np.mean(values) if values else 0
        
        # Calculate overall score (average of all metrics)
        metric_values = [v for v in avg_metrics.values() if v > 0]
        overall_score = np.mean(metric_values) if metric_values else 0
        
        agent_stats[agent_type] = {
            'avg_response_time': avg_response_time,
            'overall_score': overall_score,
            'total_records': len(data),
            'metrics': avg_metrics
        }
    
    return agent_stats


def create_evaluation_csv(agent_stats: Dict[str, Dict[str, Any]], output_dir: str):
    """Create a comprehensive CSV with all agent evaluation data"""
    
    # Prepare data for CSV
    csv_data = []
    
    for agent_name, stats in agent_stats.items():
        row = {
            'Agent_Type': agent_name,
            'Average_Response_Time_Seconds': round(stats['avg_response_time'], 3),
            'Overall_Average_Score': round(stats['overall_score'], 3),
            'Intent_Resolution_Score': round(stats['metrics'].get('intent_resolution', 0), 3),
            'Task_Adherence_Score': round(stats['metrics'].get('task_adherence', 0), 3),
            'Coherence_Score': round(stats['metrics'].get('coherence', 0), 3),
            'Fluency_Score': round(stats['metrics'].get('fluency', 0), 3),
            'Relevance_Score': round(stats['metrics'].get('relevance', 0), 3),
            'Groundedness_Score': round(stats['metrics'].get('groundedness', 0), 3),
            'Similarity_Score': round(stats['metrics'].get('similarity', 0), 3),
            'F1_Score': round(stats['metrics'].get('f1_score', 0), 3),
            'Meteor_Score': round(stats['metrics'].get('meteor_score', 0), 3),
            'Total_Records': stats['total_records']
        }
        csv_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Sort by overall score (descending)
    df = df.sort_values('Overall_Average_Score', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'agent_evaluation_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Agent evaluation CSV saved as '{csv_path}'")
    print(f"Data includes {len(df)} agents and {len(df.columns) - 2} evaluation metrics")
    
    return df


def create_response_time_chart(agent_stats: Dict[str, Dict[str, Any]], output_dir: str):
    """Create response time bar chart"""
    agents = list(agent_stats.keys())
    response_times = [agent_stats[agent]['avg_response_time'] for agent in agents]
    
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    bars = plt.bar(agents, response_times, color=colors[:len(agents)])
    
    plt.title('Average Response Time by Agent Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Agent Type', fontsize=12, fontweight='bold')
    plt.ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, response_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    chart_path = os.path.join(output_dir, 'response_time_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Response time chart saved as '{chart_path}'")


def create_overall_score_chart(agent_stats: Dict[str, Dict[str, Any]], output_dir: str):
    """Create overall score bar chart"""
    agents = list(agent_stats.keys())
    overall_scores = [agent_stats[agent]['overall_score'] for agent in agents]
    
    plt.figure(figsize=(12, 8))
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    bars = plt.bar(agents, overall_scores, color=colors[:len(agents)])
    
    plt.title('Overall Evaluation Score by Agent Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Agent Type', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Score (Average across all metrics)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, overall_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(min(overall_scores) * 0.95, max(overall_scores) * 1.05)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    chart_path = os.path.join(output_dir, 'overall_score_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overall score chart saved as '{chart_path}'")


def generate_markdown_report(agent_stats: Dict[str, Dict[str, Any]], output_path: str):
    """Generate the markdown report"""
    
    # Sort agents by overall score (descending)
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    # Sort agents by response time (ascending) for response time section
    sorted_agents_by_time = sorted(agent_stats.items(), key=lambda x: x[1]['avg_response_time'])
    
    markdown_content = """# Multi-Agent GraphRAG Evaluation Analysis Report

## Overview

This report presents a comprehensive analysis of the performance evaluation results for different agent types. The analysis covers evaluation metrics, response times, and overall performance comparisons across all evaluated companies.

## Data Sources

The detailed evaluation results for each agent are available in the corresponding JSONL files in this directory:
"""
    
    # Add data sources for each agent
    for agent_name, _ in sorted_agents:
        markdown_content += f"- `{agent_name}_evaluation_results.jsonl` - {agent_name.replace('_', ' ').title()} agent results\n"
    
    markdown_content += """
Each JSONL file contains the complete evaluation data including individual metric scores, response times, and detailed evaluation reasoning for all test questions.

## Section 1: Average Response Time Analysis

### Response Time Summary

The following table shows the average response time for each agent type:

| Agent Type   | Average Response Time (seconds) | Total Records |
|--------------|--------------------------------|---------------|
"""
    
    # Add response time table
    for agent_name, stats in sorted_agents_by_time:
        markdown_content += f"| {agent_name:<12} | {stats['avg_response_time']:>30.1f} | {stats['total_records']:>13} |\n"
    
    markdown_content += """
### Response Time Bar Chart

![Response Time by Agent](response_time_chart.png)

### Key Insights

"""
    
    # Add fastest and slowest agent insights
    fastest_agent = sorted_agents_by_time[0]
    slowest_agent = sorted_agents_by_time[-1]
    
    markdown_content += f"- **Fastest Agent**: {fastest_agent[0]} ({fastest_agent[1]['avg_response_time']:.1f} seconds)\n"
    markdown_content += f"- **Slowest Agent**: {slowest_agent[0]} ({slowest_agent[1]['avg_response_time']:.1f} seconds)\n\n"
    
    # Performance categories
    markdown_content += """### Performance Categories

Based on response times, the agents can be categorized into three performance tiers:

1. **High-Speed Tier (< 50s)**:
"""
    
    high_speed = [agent for agent, stats in sorted_agents_by_time if stats['avg_response_time'] < 50]
    for agent_name, stats in sorted_agents_by_time:
        if stats['avg_response_time'] < 50:
            markdown_content += f"   - {agent_name}: {stats['avg_response_time']:.1f}s\n"
    
    markdown_content += "\n2. **Medium-Speed Tier (50s - 200s)**:\n"
    for agent_name, stats in sorted_agents_by_time:
        if 50 <= stats['avg_response_time'] <= 200:
            markdown_content += f"   - {agent_name}: {stats['avg_response_time']:.1f}s\n"
    
    markdown_content += "\n3. **Slower Tier (> 200s)**:\n"
    for agent_name, stats in sorted_agents_by_time:
        if stats['avg_response_time'] > 200:
            markdown_content += f"   - {agent_name}: {stats['avg_response_time']:.1f}s\n"
    
    markdown_content += """

## Section 2: Overall Evaluation Metrics Analysis

### Overall Score Summary

The following table shows the overall average score across all evaluation metrics for each agent type:

| Agent Type   | Overall Score | Ranking | Total Records |
|--------------|---------------|---------|---------------|
"""
    
    # Add overall score table with ranking
    for i, (agent_name, stats) in enumerate(sorted_agents, 1):
        markdown_content += f"| {agent_name:<12} | {stats['overall_score']:>13.3f} | {i:>7} | {stats['total_records']:>13} |\n"
    
    markdown_content += """
### Overall Score Bar Chart

![Overall Score by Agent](overall_score_chart.png)

### Key Insights

"""
    
    # Add score insights
    highest_agent = sorted_agents[0]
    lowest_agent = sorted_agents[-1]
    score_range = highest_agent[1]['overall_score'] - lowest_agent[1]['overall_score']
    avg_score = np.mean([stats['overall_score'] for _, stats in sorted_agents])
    
    markdown_content += f"- **Highest Scoring Agent**: {highest_agent[0]} ({highest_agent[1]['overall_score']:.3f})\n"
    markdown_content += f"- **Lowest Scoring Agent**: {lowest_agent[0]} ({lowest_agent[1]['overall_score']:.3f})\n"
    markdown_content += f"- **Score Range**: {score_range:.3f} points\n"
    markdown_content += f"- **Average Score**: {avg_score:.3f} across all agents\n\n"
    
    markdown_content += """### Evaluation Metrics Breakdown

The overall scores are calculated by averaging across nine evaluation metrics:

1. **Intent Resolution Evaluator** - How well the agent understands and addresses user intent
2. **Task Adherence Evaluator** - How well the agent follows given instructions
3. **Coherence Evaluator** - Logical consistency and flow of responses
4. **Fluency Evaluator** - Language quality and readability
5. **Relevance Evaluator** - Appropriateness of content to the query
6. **Groundedness Evaluator** - Factual accuracy and evidence-based responses
7. **Similarity Evaluator** - Semantic similarity to expected responses
8. **F1 Score Evaluator** - Precision and recall balance
9. **Meteor Score Evaluator** - Translation quality metric adaptation

### Quality vs Speed Trade-off Analysis

Comparing overall scores with response times reveals interesting trade-offs:

"""
    
    # Add trade-off analysis
    for agent_name, stats in sorted_agents[:3]:  # Top 3 by quality
        if stats['avg_response_time'] > 500:
            speed_desc = "slowest speed"
        elif stats['avg_response_time'] < 50:
            speed_desc = "fastest speed"
        else:
            speed_desc = "moderate speed"
        
        quality_desc = "highest quality" if agent_name == sorted_agents[0][0] else "good quality"
        
        markdown_content += f"- **{agent_name}**: {quality_desc} ({stats['overall_score']:.3f}) with {speed_desc} ({stats['avg_response_time']:.1f}s)\n"
    
    markdown_content += """
The data suggests that knowledge graph agents provide superior response quality but at the cost of significantly longer processing times.

### Individual Metric Performance

| Agent Type   | Intent<br/>Resolution | Task<br/>Adherence | Coherence | Fluency | Relevance | Grounded-<br/>ness | Similarity | F1 Score | Meteor<br/>Score |
|--------------|----------------------|-------------------|-----------|---------|-----------|-------------------|------------|----------|------------------|
"""
    
    # Add individual metrics table
    for agent_name, stats in sorted_agents:
        metrics = stats['metrics']
        markdown_content += f"| {agent_name:<12} | "
        markdown_content += f"{metrics.get('intent_resolution', 0):>20.3f} | "
        markdown_content += f"{metrics.get('task_adherence', 0):>17.3f} | "
        markdown_content += f"{metrics.get('coherence', 0):>9.3f} | "
        markdown_content += f"{metrics.get('fluency', 0):>7.3f} | "
        markdown_content += f"{metrics.get('relevance', 0):>9.3f} | "
        markdown_content += f"{metrics.get('groundedness', 0):>17.3f} | "
        markdown_content += f"{metrics.get('similarity', 0):>10.3f} | "
        markdown_content += f"{metrics.get('f1_score', 0):>8.3f} | "
        markdown_content += f"{metrics.get('meteor_score', 0):>16.3f} |\n"
    
    markdown_content += """
**Metric Scale**: Intent Resolution, Task Adherence, Coherence, Fluency, Relevance, Groundedness, and Similarity are scored on a 1-5 scale. F1 Score and Meteor Score are scored on a 0-1 scale.

### Top Performers by Metric

"""
    
    # Find top performers for each metric
    metrics_list = ['intent_resolution', 'task_adherence', 'coherence', 'fluency', 'relevance', 'groundedness', 'similarity', 'f1_score', 'meteor_score']
    metric_names = ['Intent Resolution', 'Task Adherence', 'Coherence', 'Fluency', 'Relevance', 'Groundedness', 'Similarity', 'F1 Score', 'Meteor Score']
    
    for metric, display_name in zip(metrics_list, metric_names):
        best_agent = max(agent_stats.items(), key=lambda x: x[1]['metrics'].get(metric, 0))
        best_score = best_agent[1]['metrics'].get(metric, 0)
        markdown_content += f"- **{display_name}**: {best_agent[0]} ({best_score:.3f})\n"
    
    markdown_content += """
## Data Export

### Comprehensive Evaluation Data

All evaluation data presented in this report is available in a structured CSV format for further analysis, integration, and reporting purposes.

📊 **Download**: [agent_evaluation_summary.csv](agent_evaluation_summary.csv)

**File Contents:**
- Agent Type identification
- Average response time in seconds
- Overall average score across all evaluation metrics
- Individual scores for all 9 evaluation metrics:
  - Intent Resolution Score (1-5 scale)
  - Task Adherence Score (1-5 scale)
  - Coherence Score (1-5 scale)
  - Fluency Score (1-5 scale)
  - Relevance Score (1-5 scale)
  - Groundedness Score (1-5 scale)
  - Similarity Score (1-5 scale)
  - F1 Score (0-1 scale)
  - Meteor Score (0-1 scale)
- Total number of evaluation records per agent
"""
    
    # Write the markdown file
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation analysis report from JSONL files')
    parser.add_argument('files', nargs='+', help='JSONL evaluation files to process')
    parser.add_argument('--output', '-o', default='evaluation_report.md', help='Output markdown file path')
    parser.add_argument('--output-dir', '-d', default='.', help='Directory to save charts and CSV files')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the data
    print("Processing evaluation data...")
    agent_stats = process_agent_data(args.files)
    
    if not agent_stats:
        print("Error: No valid data found in the provided files")
        return
    
    print(f"Found data for {len(agent_stats)} agents")
    
    # Generate CSV
    print("Generating evaluation summary CSV...")
    create_evaluation_csv(agent_stats, args.output_dir)
    
    # Generate charts
    print("Generating charts...")
    create_response_time_chart(agent_stats, args.output_dir)
    create_overall_score_chart(agent_stats, args.output_dir)
    
    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(agent_stats, args.output)
    
    print("Report generation complete!")


if __name__ == "__main__":
    main()

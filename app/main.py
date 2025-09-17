#!/usr/bin/env python3
from __future__ import annotations
import json
"""
Azure AI Agent Team for GraphRAG Financial Analysis

This module implements a multi-agent system using Azure AI Agents that combines:
- GraphRAG (Graph-based Retrieval Augmented Generation) for knowledge graph queries
- Azure AI Search for document retrieval
- Bing Search for external information grounding

The system consists of three specialized agents:
1. TeamLeader: Orchestrates tasks between specialized agents
2. RAG-agent: Performs document search using Azure AI Search
3. KG-agent: Queries knowledge graphs using GraphRAG
4. Bing-agent: Provides external information through Bing Search

USAGE:
    python agent.py

REQUIREMENTS:
    - Azure AI Projects subscription
    - Azure AI Agents enabled
    - OpenAI API access
    - GraphRAG index files in ./apple/output/
    - Environment variables: OPENAI_API_KEY, AZURE_* credentials

Environment Variables:
    AZURE_OPENAI_API_KEY: Azure OpenAI API key for language models
"""

"""
ENVIRONMENT VARIABLES CONFIGURATION:

All sensitive information and configuration values have been moved to environment variables.
The following variables are supported in the .env file:

REQUIRED VARIABLES:
- AZURE_OPENAI_API_KEY: Azure OpenAI API key for language models
- API_KEY: API key for securing FastAPI endpoints (auto-generated if not provided)

OPTIONAL VARIABLES:
- ENV_FILE_PATH: Path to .env file (default: "/Users/shanepeckham/sources/graphrag/apple/.env")
- PROJECT_ENDPOINT: Azure AI Project endpoint (default: Azure fiagent endpoint)
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint (default: Azure fiagent cognitive services)
- MODEL_DEPLOYMENT_NAME: Main model deployment name (default: "gpt-4.1")
- REASONING_MODEL_DEPLOYMENT_NAME: Reasoning model deployment name (default: "o3-mini")
- AI_SEARCH_TYPE: Search type (default: "SIMPLE")
- GRAPH_QUERY_TYPE: Graph query type (default: "local")
- AI_SEARCH_INDEX_NAME: AI Search index name (default: "apple_report_agent")
- BING_CONNECTION_NAME: Bing Search connection name (default: "agentbing")
- RAW_INPUT_PATH: Raw input data path
- OUTPUT_PATH: Output data path
- GRAPH_OUTPUT_PATH: Graph output data path
- INPUT_DIR: Input directory for GraphRAG (default: "./apple/output")


"""

# pylint: disable=line-too-long,useless-suppression
# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------



# Core imports
import asyncio
import logging
import hashlib
import os
import secrets
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import yaml

# Python 3.9+ type annotations
if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Set
else:
    Dict = dict
    List = list
    Tuple = tuple
    Set = set
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Suppress Pydantic v1/v2 compatibility warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ForwardRef.*")

# Compatibility fix for spaCy + Pydantic v2
os.environ["SPACY_WARNING_IGNORE"] = "W007"

# Import order is critical - import these first
try:
    import pydantic
    import typing_extensions
    # Check for TypeIs availability
    try:
        from typing_extensions import TypeIs
        logging.info("✅ TypeIs imported successfully")
    except ImportError:
        logging.warning("⚠️  TypeIs not available, using workaround")
        # Create a dummy TypeIs for compatibility
        if TYPE_CHECKING:
            from typing_extensions import TypeIs
        else:
            def _type_is_workaround(x):
                return x
            TypeIs = _type_is_workaround

    # Force import of the correct version
    if not pydantic.__version__.startswith("2."):
        msg = f"Wrong Pydantic version: {pydantic.__version__}"
        raise ImportError(msg)

except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Please install: pip install 'pydantic>=2.5.0' 'typing-extensions>=4.9.0'")
    raise

import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Azure AI and Identity imports
from azure.ai.agents.models import (
    FunctionTool, ToolSet, AsyncFunctionTool, AsyncToolSet,
    AzureAISearchQueryType, AzureAISearchTool, BingGroundingTool, BingGroundingSearchConfiguration
)
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# GraphRAG imports
import graphrag.api as api
from graphrag.cli.query import _resolve_output_files
from graphrag.config.enums import ModelType
from graphrag.config.load_config import load_config
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates, read_indexer_entities, read_indexer_relationships,
    read_indexer_reports, read_indexer_text_units, read_indexer_communities,
    read_indexer_report_embeddings
)
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Local utility imports
import sys
from pathlib import Path

from agent_team_dashboard import AgentTeam, AgentTask, emit_kg_sources_update
from agent_trace_configurator import AgentTraceConfigurator

# Conditional import for WebSocket manager
try:
    from websocket_manager import websocket_manager
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    print(f"WebSocket manager not available: {e}")
    websocket_manager = None
    WEBSOCKET_AVAILABLE = False

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, WebSocket, WebSocketDisconnect, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from opentelemetry import trace
tracer = trace.get_tracer(__name__)

from contextlib import asynccontextmanager

# Global variables to store loaded data
_graphrag_data = None
_language_models = None
_project_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    print("🚀 Starting Financial Analysis Agent Team...")

    # Load all heavy resources once
    global _language_models, _project_client

    try:
        # Initialize configuration once
        _setup_tiktoken_for_gpt41()
        _setup_logging()
        _load_environment_variables()

        # Initialize language models once
        print("📚 Initializing language models...")
        _language_models = _initialize_language_models()

        # Initialize Azure project client once
        print("☁️ Initializing Azure client...")
        _project_client = _create_azure_project_client()

        # Load GraphRAG data once
        #print("🔍 Loading GraphRAG data...")
        #_graphrag_data = _load_graphrag_data()

        print("✅ Application startup complete!")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    print("🛑 Shutting down application...")
    if _project_client:
        _project_client.close()
    print("✅ Shutdown complete!")

class QueryResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    response: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    query: str
    context: str
    thread_id: str
    run_id: str  # Used in evaluation mode to track a single agent
    token_usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage statistics for all agents")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")
    graph_query_type: str = Field(..., description="Query method: global, local, drift, basic")
    search_query_type: str = Field(..., description="Search type: SIMPLE, SEMANTIC")
    use_search: bool = Field(
        default=True,
        description="Whether to use Azure AI Search for document retrieval"
    )
    use_graph: bool = Field(
        default=True,
        description="Whether to use GraphRAG for knowledge graph queries"
    )
    use_web: bool = Field(
        default=False,
        description="Whether to use Bing Search for external information grounding"
    )
    use_reasoning: bool = Field(
        default=False,
        description="Whether to use reasoning capabilities for complex queries"
    )
    evaluation_mode: bool = Field(
        default=False,
        description="Whether to run in evaluation mode (no WebSocket updates)"
    )

# FastAPI app initialization
app = FastAPI(
    title="FI-Multi-Agent Team with GraphRAG",
    description="REST API for FI-Multi-Agent Team with GraphRAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Add lifespan handler
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# CONFIGURATION AND INITIALIZATION
# ==============================================================================

def _setup_tiktoken_for_gpt41() -> None:
    """
    Register the tokenizer for GPT-4.1 to fix tokenization errors.

    GPT-4.1 is not natively supported by tiktoken, so we register it
    to use the cl100k_base encoding which is compatible.
    """
    try:
        tiktoken.encoding_for_model("gpt-4.1")
    except KeyError:
        # Register gpt-4.1 to use cl100k_base encoding
        tiktoken.get_encoding("cl100k_base")
        tiktoken.model.MODEL_TO_ENCODING["gpt-4.1"] = "cl100k_base"

def _setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(level=logging.ERROR)

    # Configure Azure resource logger
    logger = logging.getLogger('azure.mgmt.resource')
    logger.setLevel(logging.ERROR)

def _load_environment_variables() -> None:
    """
    Load environment variables from .env file and verify required keys.

    Raises:
        FileNotFoundError: If .env file is not found in the script directory
        ValueError: If required environment variables are missing
    """
    # Get the directory where main.py is located
    script_dir = Path(__file__).parent
    env_file_path = script_dir / ".env"

    # Check if .env file exists
    if not env_file_path.exists():
        raise FileNotFoundError(f".env file not found at {env_file_path}. Please create a .env file in the same directory as main.py")

    # Load environment variables from .env file
    load_dotenv(env_file_path)

    # Verify critical environment variables
    required_vars = ["AZURE_OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    print(f"AZURE_OPENAI_API_KEY present: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
    print(f"PROJECT_ENDPOINT present: {bool(os.getenv('PROJECT_ENDPOINT'))}")
    print(f"AZURE_OPENAI_ENDPOINT present: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")

# Initialize configuration
_setup_tiktoken_for_gpt41()
_setup_logging()
_load_environment_variables()

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# File paths and directories
RAW_INPUT_PATH = os.getenv("RAW_INPUT_PATH", "/Users/shanepeckham/sources/data//extracted_text")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/Users/shanepeckham/sources/data/data/processed_text")
INPUT_PATH = OUTPUT_PATH
GRAPH_OUTPUT_PATH = os.getenv("GRAPH_OUTPUT_PATH", "/Users/shanepeckham/sources/data/data/graph_output/")
INPUT_DIR = os.getenv("INPUT_DIR", "./data/output")
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

# Dataset and API configuration
DATASET_DESCRIPTION = "Report"
API_VERSION = "2024-02-15-preview"

# GraphRAG table names
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2

# Azure AI configuration
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1")
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT", "https://fiagent-resource.services.ai.azure.com/api/projects/fiagent/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://fiagent-resource.cognitiveservices.azure.com/")
REASONING_MODEL_DEPLOYMENT_NAME = os.getenv("REASONING_MODEL_DEPLOYMENT_NAME", "o3-mini")

# Query configuration
AI_SEARCH_TYPE = os.getenv("AI_SEARCH_TYPE", "SIMPLE")
AI_SEARCH_CONNECTION_NAME = os.getenv("AI_SEARCH_CONNECTION_NAME", "agentsearcher")
GRAPH_QUERY_TYPE = os.getenv("GRAPH_QUERY_TYPE", "local")
AI_SEARCH_INDEX_NAME = os.getenv("AI_SEARCH_INDEX_NAME", "report_agent")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME", "agentbing")
BING_CONFIGURATION = os.getenv("BING_CONFIGURATION", "sky")

# TEAM configuration
TEAM_NAME = os.getenv("TEAM_NAME", "cr_team")
TEAM_DESCRIPTION = "A team of agents specialized in financial analysis and reporting."

# Sample questions for testing
SAMPLE_QUESTIONS = [
    "Are there any mine safety disclosures?",
    "What were the net sales by reportable segment for 2024, 2023 and 2022 for Japan?",
    "What is the percentage change for accessories net sales between 2023 and 2024?",
    "What would be the five-year cumulative total shareholder return if $100 was invested on September 2019 on the S&P 500 index?",
    "Given only the information provided to you, with no public record searches, evaluate the financial health of the company. What are the key indicators of financial health? What are the key risks to financial health? What are the key opportunities for financial health? What is your overall assessment of the company's financial health? Would you invest in this company? Why or why not?"
    "Are there any drops in revenue? If yes, what are the reasons for the drop? Which services/products are affected? What is the percentage drop in revenue? ",
    "How many shareholders were present in 18 October 2024?"
]

# Current question being processed
CURRENT_QUESTION = "Are there any drops in revenue? If yes, what are the reasons for the drop? Which services/products are affected? What is the percentage drop in revenue? "

REASON_CURRENT_QUESTION = (
        "Given only the information provided to you, evaluate the financial health of the company. "
        "What are the key indicators of financial health? "
        "What are the key risks to financial health? What are the key opportunities for financial health? "
        "What is your overall assessment of the company's financial health? Would you invest in this company? "
        "Why or why not?"
    )
EVALUATION_MODE = False
# ==============================================================================
# API SECURITY CONFIGURATION
# ==============================================================================

# Load API key from environment first
API_KEY = os.getenv("API_KEY")

# Generate a secure API key only if not provided in environment
if not API_KEY:
    API_KEY = "graphrag_" + secrets.token_urlsafe(32)
    print(f"🔑 No API_KEY found in environment. Generated new API key: {API_KEY}")
    print("💡 To use a fixed API key, set API_KEY in your environment variables")
else:
    print(f"🔑 API_KEY loaded from environment: {API_KEY[:20]}...{API_KEY[-4:]}")

# Hash the API key for secure comparison
API_KEY_HASH = hashlib.sha256(API_KEY.encode()).hexdigest()
print(f"🔒 API key hash: {API_KEY_HASH[:16]}...")

# Security scheme for FastAPI
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """
    Verify the provided API key against the stored hash.

    Args:
        credentials: The Bearer token credentials from the request header

    Returns:
        bool: True if the API key is valid

    Raises:
        HTTPException: If the API key is invalid or missing
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Hash the provided token for comparison
    provided_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()

    # Compare hashes to prevent timing attacks
    if not secrets.compare_digest(API_KEY_HASH, provided_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True

# ==============================================================================
# AZURE AI CLIENT INITIALIZATION
# ==============================================================================

def _create_azure_project_client() -> AIProjectClient:
    """
    Create and return an Azure AI Project client.

    Returns:
        AIProjectClient: Configured Azure AI Project client
    """
    return AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

# Azure project client will be initialized in lifespan function

# ==============================================================================
# AGENT TASK MANAGEMENT FUNCTIONS
# ==============================================================================
@tracer.start_as_current_span("create_task")  # type: ignore
def create_task(recipient: str, request: str, requestor: str) -> str:
    """
    Request another agent in the team to complete a task.

    This function facilitates inter-agent communication within an AgentTeam by
    allowing one agent to delegate tasks to another specialized agent.

    Args:
        team_name: The name of the team containing the agents
        recipient: The name of the target agent to receive the task
        request: Description of the task to complete (can be a question or instruction)
        requestor: The name of the agent making the request

    Returns:
        str: "True" if the task was successfully queued, "False" otherwise

    Example:
        >>> create_task("finance_team", "KG-agent", "Find revenue data for Q4", "TeamLeader")
        "True"
    """
    task = AgentTask(recipient=recipient, task_description=request, requestor=requestor)
    team: Optional[AgentTeam] = None

    try:
        team = AgentTeam.get_team(TEAM_NAME)
    except Exception:
        # Log the exception if needed, but continue gracefully
        pass

    if team is not None:
        team.add_task(task)
        return "True"
    return "False"

# ==============================================================================
# LANGUAGE MODEL CONFIGURATION
# ==============================================================================

def _create_chat_model_config() -> LanguageModelConfig:
    """
    Create configuration for the chat language model.

    Returns:
        LanguageModelConfig: Configuration for GPT-4.1 chat model
    """
    return LanguageModelConfig(
        api_key=AZURE_OPENAI_API_KEY,
        type=ModelType.AzureOpenAIChat,
        deployment_name=MODEL_DEPLOYMENT_NAME,
        api_version="2025-01-01-preview",
        model=MODEL_DEPLOYMENT_NAME,
        max_retries=20,
        api_base=AZURE_OPENAI_ENDPOINT,
    )

def _create_embedding_model_config() -> LanguageModelConfig:
    """
    Create configuration for the embedding model.

    Returns:
        LanguageModelConfig: Configuration for text-embedding-ada-002 model
    """
    return LanguageModelConfig(
        api_key=AZURE_OPENAI_API_KEY,
        type=ModelType.AzureOpenAIEmbedding,
        api_version="2023-05-15",
        deployment_name="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_base=AZURE_OPENAI_ENDPOINT,
        max_retries=20,
    )

def _initialize_language_models() -> Tuple[Any, Any, Any]:
    """
    Initialize chat model, embedding model, and tokenizer.

    Returns:
        Tuple[Any, Any, Any]: (chat_model, text_embedder, encoder)
    """
    # Create model configurations
    chat_config = _create_chat_model_config()
    embedding_config = _create_embedding_model_config()

    # Initialize models through ModelManager
    model_manager = ModelManager()

    chat_model = model_manager.get_or_create_chat_model(
        name="search",
        model_type=ModelType.AzureOpenAIChat,
        config=chat_config,
    )

    text_embedder = model_manager.get_or_create_embedding_model(
        name="search_embedding",
        model_type=ModelType.AzureOpenAIEmbedding,
        config=embedding_config,
    )

    # Initialize tokenizer - using o200k_base for GPT-4 models
    encoder = tiktoken.get_encoding("o200k_base")

    return chat_model, text_embedder, encoder

# Note: Language models and GraphRAG data are now initialized in the lifespan function
# Global variables will hold the initialized resources

# ==============================================================================
# GRAPHRAG DATA LOADING AND INITIALIZATION
# ==============================================================================


def read_community_reports(input_dir: str, community_report_table: str = COMMUNITY_REPORT_TABLE) -> pd.DataFrame:
    """
    Read community reports from parquet file.

    Args:
        input_dir: Directory containing the parquet files
        community_report_table: Name of the community report table

    Returns:
        pd.DataFrame: Community reports dataframe

    Raises:
        FileNotFoundError: If the parquet file doesn't exist
    """
    input_path = Path(input_dir) / f"{community_report_table}.parquet"
    return pd.read_parquet(input_path)

# ==============================================================================
# AGENT TEAM CONFIGURATION AND MAIN EXECUTION
# ==============================================================================

def _create_agent_toolsets() -> Tuple[ToolSet, AsyncToolSet]:
    """
    Create toolsets for sync and async functions.

    Returns:
        Tuple[ToolSet, AsyncToolSet]: (sync_toolset, async_toolset)
    """
    # Define functions available to agents
    agent_team_default_functions: set = {
        create_task,
    }

    # Create sync and async toolsets
    sync_functions = FunctionTool(functions=agent_team_default_functions)
    #async_functions = AsyncFunctionTool()

    sync_toolset = ToolSet()
    sync_toolset.add(sync_functions)

    async_toolset = AsyncToolSet()
    #async_toolset.add(async_functions)

    return sync_toolset, None

def _setup_agent_team_with_globals(question: str, search_query_type: str, graph_query_type: str, use_search: bool, use_graph: bool, use_web: bool, use_reasoning: bool, evaluation_mode: bool) -> str:
    """
    Set up and run the agent team using pre-loaded global resources.

    This function uses resources loaded at startup to avoid reloading heavy data.
    """
    global _language_models, _project_client

    if not _project_client:
        return "Error: Azure project client not initialized"

    if evaluation_mode:
        print("Running in evaluation mode, no WebSocket updates will be sent")
        EVALUATION_MODE = True
        WEBSOCKET_EVENTS_ENABLED = False

    # Use the shared agents client without 'with' statement to keep it open
    agents_client = _project_client.agents

    # Create tools and toolsets using pre-loaded data
    sync_toolset, _ = _create_agent_toolsets()

    # Register all agent functions
    agents_client.enable_auto_function_calls({create_task, fetch_weather})

    if MODEL_DEPLOYMENT_NAME is not None:
        # Setup tracing for debugging
        AgentTraceConfigurator(agents_client=agents_client).setup_tracing()

        # Create agent team without using 'with' statement to avoid closing the client
        agent_team = AgentTeam(TEAM_NAME, agents_client=agents_client)

        # Get the directory of the current script to ensure we find the config file
        script_dir = Path(__file__).parent
        config_file_path = script_dir / "agent_team_config.yaml"

        print(f"📁 Script directory: {script_dir}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Looking for config file at: {config_file_path}")
        print(f"📁 Config file exists: {config_file_path.exists()}")

        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS = config["TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS"].strip()
            TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS = config["TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS"].strip()
            WEATHER_AGENT_DESCRIPTION = config["WEATHER_AGENT_DESCRIPTION"].strip()
            WEATHER_AGENT_INSTRUCTIONS = config["WEATHER_AGENT_INSTRUCTIONS"].strip()
            CLASSIFIER_AGENT_DESCRIPTION = config["CLASSIFIER_AGENT_DESCRIPTION"].strip()
            CLASSIFIER_AGENT_INSTRUCTIONS = config["CLASSIFIER_AGENT_INSTRUCTIONS"].strip()

            if not use_reasoning:
                TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS += f"\n\n{WEATHER_AGENT_DESCRIPTION}"
                TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS += f"\n\n{CLASSIFIER_AGENT_DESCRIPTION}"
            else:
                TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS += f"\n\n{WEATHER_AGENT_DESCRIPTION}"
                TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS += f"\n\n{CLASSIFIER_AGENT_DESCRIPTION}"

                # If no question is provided, use the reasoning current question
                if question == "":
                    question = REASON_CURRENT_QUESTION

        # Configure Team Leader (simplified configuration)
        if not use_reasoning:
            agent_team.set_team_leader(
                model=MODEL_DEPLOYMENT_NAME,
                name="TeamLeader",
                instructions=(TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS),
                toolset=sync_toolset,
            )
        else:
            # Reasoning agent uses a different model
            agent_team.set_team_leader(
                model=REASONING_MODEL_DEPLOYMENT_NAME,
                name="TeamLeader",
                instructions=(TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS),
                toolset=sync_toolset,
            )

        # Configure agents with proper toolsets
        fetch_weather_tool = FunctionTool(functions={fetch_weather})
        user_toolset = ToolSet()
        user_toolset.add(fetch_weather_tool)
        agent_team.add_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="Weather-agent-multi",
            instructions=(WEATHER_AGENT_INSTRUCTIONS),
            tools=user_toolset.definitions,
            can_delegate=False
        )

        agent_team.add_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="Classifier-agent-multi",
            instructions=(CLASSIFIER_AGENT_INSTRUCTIONS),
            can_delegate=False
        )

        # Assemble and run the team
        print("🔧 Assembling agent team...")
        agent_team.assemble_team()

        print(f"🚀 Starting agent team processing for question: {question}")
        print(f"📊 Team configuration:")
        print(f"   - Use Search: {use_search}")
        print(f"   - Use Graph: {use_graph} (type: {graph_query_type})")
        print(f"   - Use Web: {use_web}")
        print(f"   - Use Reasoning: {use_reasoning}")
        print(f"   - Evaluation: {evaluation_mode}")

        # Process the request and ensure we wait for completion
        result = agent_team.process_request(request=question, evaluation_mode=evaluation_mode)
        agent_team.dismantle_team()

        print(f"✅ Agent team processing completed")
        print(f"📝 Result length: {len(result) if result else 0} characters")

        if not result:
            print("⚠️  Error: Agent team returned empty or incomplete response. Please try again.")
            return "Error: Agent team returned empty or incomplete response. Please try again."

        return result

    return "Error: MODEL_DEPLOYMENT_NAME is not set"


def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location: The location to fetch weather for.
    :return: Weather information as a JSON string.
    """
    # Mock weather data for demonstration purposes
    mock_weather_data = {"New York": "Sunny, 25°C", "London": "Cloudy, 18°C", "Tokyo": "Rainy, 22°C"}
    weather = mock_weather_data.get(location, "Weather data not available for this location.")
    response = {
        "text": weather,
    }
    return weather


def _create_search_tools_with_type(project_client: AIProjectClient, search_type: str) -> Tuple[AzureAISearchTool, BingGroundingTool]:
    """Create search tools with specified search type."""
    # Azure AI Search tool
    search_conn = project_client.connections.get(name=AI_SEARCH_CONNECTION_NAME, include_credentials=True)

    if search_type == "SEMANTIC":
        search_tool = AzureAISearchTool(
            index_connection_id=search_conn.id,
            index_name=AI_SEARCH_INDEX_NAME,
            query_type=AzureAISearchQueryType.SEMANTIC,
            top_k=3,
            filter=""
        )
    elif search_type == "HYBRID":
        search_tool = AzureAISearchTool(
            index_connection_id=search_conn.id,
            index_name=AI_SEARCH_INDEX_NAME,
            query_type=AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,  # Use semantic for hybrid (combines vector + keyword)
            top_k=3,
            filter=""
        )
    else:  # Default to SIMPLE
        search_tool = AzureAISearchTool(
            index_connection_id=search_conn.id,
            index_name=AI_SEARCH_INDEX_NAME,
            query_type=AzureAISearchQueryType.SIMPLE,
            top_k=3,
            filter=""
        )

    # Bing Search tool
    bing_conn = project_client.connections.get(name=BING_CONNECTION_NAME, include_credentials=True)
    bing_tool = BingGroundingTool(connection_id=bing_conn.id)

    return search_tool, bing_tool



@app.get("/")
async def root():
    """Root endpoint with API information (no authentication required)."""
    return {
        "message": "GraphRAG API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "querying": {
                "/query_team": "POST - Query the agent team",
            },
            "utilities": {
                "/health": "GET - Health check",
            }
        },
        "query_parameters": {
            "query": "string - The question to ask",
            "search_query_type": "string - SIMPLE or SEMANTIC",
            "graph_query_type": "string - global, local, drift, or basic",
            "use_search": "boolean - Enable Azure AI Search",
            "use_graph": "boolean - Enable GraphRAG",
            "use_web": "boolean - Enable Bing Search",
            "use_reasoning": "boolean - Enable reasoning mode",
            "evaluation_mode": "boolean - Disable WebSocket updates for evaluation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "GraphRAG API"}

@app.get("/api/config/agent_team_config.yaml")
async def get_agent_config():
    """Get the agent team configuration YAML file."""
    try:
        config_file_path = Path(__file__).parent / "agent_team_config.yaml"

        if not config_file_path.exists():
            raise HTTPException(status_code=404, detail="Configuration file not found")

        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_content = file.read()

        return Response(
            content=config_content,
            media_type="text/plain",
            headers={"Content-Disposition": "inline; filename=agent_team_config.yaml"}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading configuration: {str(e)}")

@app.post("/api/config/agent_team_config.yaml")
async def save_agent_config(request: Request):
    """Save the agent team configuration YAML file."""
    try:
        config_file_path = Path(__file__).parent / "agent_team_config.yaml"

        # Read the request body as text
        config_content = await request.body()
        config_text = config_content.decode('utf-8')

        if not config_text.strip():
            raise HTTPException(status_code=400, detail="Configuration content cannot be empty")

        # Validate YAML syntax
        try:
            yaml.safe_load(config_text)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML syntax: {str(e)}")

        # Create backup of existing file
        if config_file_path.exists():
            backup_path = config_file_path.with_suffix(f".yaml.backup.{int(time.time())}")
            config_file_path.rename(backup_path)
            print(f"Created backup: {backup_path}")

        # Write new configuration
        with open(config_file_path, 'w', encoding='utf-8') as file:
            file.write(config_text)

        return {"message": "Configuration saved successfully", "path": str(config_file_path)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")



@app.post("/query_team", response_model=QueryResponse)
def query_team_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Query the agent team for financial analysis.

    This endpoint uses pre-loaded resources from application startup
    to provide fast responses without reloading data. No authentication required.
    """
    try:
        # Use global variables loaded at startup
        global _language_models, _project_client, _request
        _request = request

        if not all([_language_models, _project_client]):
            raise HTTPException(status_code=503, detail="Application not fully initialized")

        question = request.query
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Query is required")

        # Run the agent team with the question using pre-loaded resources
        markdown_response, context, thread_id, run_id, token_usage = _setup_agent_team_with_globals(question, request.search_query_type, request.graph_query_type, use_search=request.use_search,
                                                           use_graph=False, use_web=request.use_web, use_reasoning=request.use_reasoning, evaluation_mode=request.evaluation_mode)

        return QueryResponse(
            response=markdown_response,
            query=question,
            context=context,
            thread_id=thread_id,
            run_id=run_id,
            token_usage=token_usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==============================================================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME VISUALIZATION
# ==============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent team visualization"""
    if not WEBSOCKET_AVAILABLE or not websocket_manager:
        await websocket.close(code=1011, reason="WebSocket not available")
        return

    await websocket_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep the connection alive and listen for messages
            data = await websocket.receive_text()
            # Echo back or handle specific messages if needed
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, session_id)


@app.get("/dashboard")
async def get_dashboard():
    """Serve the real-time dashboard"""
    dashboard_path = Path(__file__).parent.parent / "UI" / "realtime_dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_path)


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Run the GraphRAG Agent Team application.

    This function initializes and runs the multi-agent financial analysis system.
    """
    try:
        import uvicorn
        logging.info("Starting Financial Analysis Agent Team with GraphRAG...")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        logging.exception("Error running agent team: %s", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()

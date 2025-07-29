#!/usr/bin/env python3
from __future__ import annotations
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
    AzureAISearchQueryType, AzureAISearchTool, BingGroundingTool, BingGroundingSearchConfiguration, BingCustomSearchTool
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
    global _graphrag_data, _language_models, _project_client
    
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
        print("🔍 Loading GraphRAG data...")
        _graphrag_data = _load_graphrag_data()
        
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

def _load_graphrag_data() -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Load all required GraphRAG data from parquet files.
    
    Returns:
        Tuple containing: (entities, relationships, reports, text_units, 
                          communities, description_embedding_store, full_content_embedding_store,
                          covariates)

    Raises:
        FileNotFoundError: If required parquet files are missing
        Exception: If data loading fails
    """
    try:
        # Load dataframes from parquet files
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        
        # Process data using GraphRAG indexer adapters
        entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
        relationships = read_indexer_relationships(relationship_df)
        text_units = read_indexer_text_units(text_unit_df)
        communities = read_indexer_communities(community_df, report_df)
        # Load covariates if available
        try:
            covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
            claims = read_indexer_covariates(covariate_df)
            covariates = {"claims": claims}
            print(f"Loading covariates from {COVARIATE_TABLE}.parquet in {INPUT_DIR}")
        except FileNotFoundError:
            print(f"No covariates found in {COVARIATE_TABLE}.parquet in {INPUT_DIR}")
            covariates = None

        # Process reports with embeddings
        reports = read_indexer_reports(
            report_df,
            community_df,
            COMMUNITY_LEVEL,
            content_embedding_col="full_content_embeddings",
        )
        
        # Initialize vector stores for embeddings
        description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description",
        )
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        
        full_content_embedding_store = LanceDBVectorStore(
            collection_name="default-community-full_content",
        )
        full_content_embedding_store.connect(db_uri=LANCEDB_URI)
        
        # Load report embeddings into vector store
        read_indexer_report_embeddings(reports, full_content_embedding_store)
        
        return (entities, relationships, reports, text_units, communities,
                description_embedding_store, full_content_embedding_store, covariates)
        
    except Exception as e:
        print(f"Error loading GraphRAG data: {e}")
        

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

# Load GraphRAG data
# Note: This will be initialized in the lifespan function, not at module level

# ==============================================================================
# GRAPHRAG SEARCH IMPLEMENTATIONS
# ==============================================================================

async def _query_graph_async_drift(question: str) -> str:
    """
    Execute a DRIFT search query against the knowledge graph.
    
    DRIFT (Dynamic Retrieval and Information Focused Targeting) search provides
    advanced query capabilities with multiple levels of analysis.
    
    Args:
        question: The question to search for in the knowledge graph
        
    Returns:
        str: The search response
        
    Raises:
        Exception: If the search operation fails
    """
    global _graphrag_data, _language_models
    
    if not _graphrag_data or not _language_models:
        raise RuntimeError("GraphRAG data or language models not initialized")
    
    chat_model, text_embedder, encoder = _language_models
    (entities, relationships, reports, text_units, communities,
     description_embedding_store, full_content_embedding_store, covariates) = _graphrag_data
    
    print(f"GRAPH: Using drift search with question: {question}")
    drift_params = DRIFTSearchConfig(
        primer_folds=1,
        drift_k_followups=3,
        n_depth=3,
    )

    context_builder = DRIFTSearchContextBuilder(
        model=chat_model,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=reports,
        entity_text_embeddings=description_embedding_store,
        text_units=text_units,
        token_encoder=encoder,
        config=drift_params,
    )

    search = DRIFTSearch(
        model=chat_model, 
        context_builder=context_builder, 
        token_encoder=encoder
    )

    result = await search.search(question)
    print(result.response)
    _kg_sources = parse_graphrag_metadata(result.response)
    return str(result.response)

async def _query_graph_async_global(question: str) -> str:
    """
    Execute a global search query against the knowledge graph.
    
    Global search analyzes community reports and provides high-level insights
    across the entire knowledge graph using map-reduce methodology.
    
    Args:
        question: The question to search for in the knowledge graph
        
    Returns:
        str: The search response
        
    Raises:
        Exception: If the search operation fails
    """
    global _graphrag_data, _language_models
    
    if not _graphrag_data or not _language_models:
        raise RuntimeError("GraphRAG data or language models not initialized")
    
    chat_model, text_embedder, encoder = _language_models
    (entities, relationships, reports, text_units, communities,
     description_embedding_store, full_content_embedding_store, covariates) = _graphrag_data
    
    print(f"GRAPH: Using global search with question: {question}")
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=encoder,
    )
    
    context_builder_params = {
        "use_community_summary": False,  # Use full community reports
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        model=chat_model,
        context_builder=context_builder,
        token_encoder=encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )
    
    result = await search_engine.search(question)
    print(result.response)
    _kg_sources = parse_graphrag_metadata(result.response)
    return str(result.response)

async def _query_graph_async_local(question: str) -> str:
    """
    Execute a local search query against the knowledge graph.
    
    Local search focuses on specific entities and their immediate relationships,
    providing detailed information about particular aspects of the data.
    
    Args:
        question: The question to search for in the knowledge graph
        
    Returns:
        str: The search response
        
    Raises:
        Exception: If the search operation fails
    """
    global _graphrag_data, _language_models 
    
    if not _graphrag_data or not _language_models:
        raise RuntimeError("GraphRAG data or language models not initialized")
    
    chat_model, text_embedder, encoder = _language_models
    (entities, relationships, reports, text_units, communities,
     description_embedding_store, full_content_embedding_store, covariates) = _graphrag_data
 
    print(f"GRAPH: Using local search with question: {question}")
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,  # Set to None if covariates weren't generated during indexing
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    model_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    search_engine = LocalSearch(
        model=chat_model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params=model_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )
    
    result = await search_engine.search(question)
    print(result.response)
    _kg_sources = parse_graphrag_metadata(result.response)

    return str(result.response)

def parse_graphrag_metadata(response: str) -> dict:
    """
    Parse GraphRAG response to extract metadata about sources, entities, and relationships.
    
    Searches for patterns like:
    - [Data: Sources (119)]
    - [Entities: 5135, 1555]
    - [Relationships: 54421, 35035]
    - Combined: [Data: Sources (119) ([Entities: 5135, 1555]; [Relationships: 54421, 35035])
    
    Args:
        response: The GraphRAG response string
        
    Returns:
        dict: Parsed metadata with keys 'Sources', 'Entities', 'Relationships'
              Example: {Sources:[119], Entities:[5135, 1555], Relationships: [54421, 35035]}
    """
    import re
    
    # Initialize result dictionary
    result = {
        "Sources": [],
        "Entities": [],
        "Relationships": []
    }
    
    try:
        # Pattern 1: Sources - matches "[Data: Sources (2305, 2603)]"
        sources_pattern = r'Data:\s*Sources\s*\(([^)]+)\)'
        sources_match = re.search(sources_pattern, response)
        if sources_match:
            sources_str = sources_match.group(1).strip()
            # Extract all numbers from the sources
            sources = [int(x.strip()) for x in sources_str.split(',') if x.strip().isdigit()]
            result["Sources"] = sources
            print(f"🔍 Found Sources: {sources}")
        
        # Pattern 2: Entities - matches "[Data: Entities (4264, 661, 760)]"
        entities_pattern = r'Data:\s*Entities\s*\(([^)]+)\)'
        entities_match = re.search(entities_pattern, response)
        if entities_match:
            entities_str = entities_match.group(1).strip()
            # Extract all numbers from the entities (ignore non-numeric parts)
            entities = [int(x.strip()) for x in entities_str.split(',') if x.strip().isdigit()]
            result["Entities"] = entities
            print(f"🔍 Found Entities: {entities}")
        
        # Pattern 3: Relationships - matches "[Data: Relationships (55392, 84934, 75511, +more)]"
        relationships_pattern = r'Relationships\s*\(([^)]+)\)'
        relationships_match = re.search(relationships_pattern, response)
        if relationships_match:
            relationships_str = relationships_match.group(1).strip()
            # Extract all numbers from relationships (ignore "+more" and other non-numeric parts)
            relationships = [int(x.strip()) for x in relationships_str.split(',') if x.strip().isdigit()]
            result["Relationships"] = relationships
            print(f"🔍 Found Relationships: {relationships}")
        
        # Summary
        found_items = [key for key, value in result.items() if value]
        

        def get_text_units(text_units: list) -> list:
            """
            Helper function to get text units from the text_units list.
            """
            texts = []
            for id in text_units:
                for v in _graphrag_data[3]: # Text Units
                    if v.id == str(id):
                        texts.append(v.text)
            return texts

        sources = {}
        for key, value in result.items():
            if key == "Entities":
                for id in value:
                    for v in _graphrag_data[0]: # Entities
                        if v.short_id == str(id):
                            texts = get_text_units(v.text_unit_ids)
                            sources[id] = texts

            if key == "Relationships":
                for id in value:
                    for v in _graphrag_data[1]: # Relationships
                        if v.short_id == str(id):
                            texts = get_text_units(v.text_unit_ids)
                            sources[id] = texts
            
            if key == "Sources":
                for id in value:
                    for i in range(len(_graphrag_data)):
                        try:
                            for v in _graphrag_data[3]: # Text Units
                                if v.short_id == str(id):
                                    sources[id] = [v.text]

                        except Exception as e:
                            continue

        if found_items:
            print(f"📊 Parsed GraphRAG metadata: {result}")
            # Emit WebSocket update for the dashboard with source texts
            emit_kg_sources_update({
                "Sources": result.get("Sources", []),
                "Entities": result.get("Entities", []),
                "Relationships": result.get("Relationships", []),
                "source_texts": sources  # Include the actual source texts
            })
        else:
            print("⚠️  No GraphRAG metadata found in response")

    except Exception as e:
        print(f"❌ Error parsing GraphRAG metadata: {e}")
    
    return sources

@tracer.start_as_current_span("query_graph")  # type: ignore
def query_graph(question: str, search_type: str = "local") -> str:
    """
    Query the GraphRAG knowledge graph with thread-safe async execution.
    
    This function provides a synchronous interface to the async GraphRAG search
    capabilities, handling event loop management to work properly within the
    Azure AI Agents framework.
    
    Args:
        question: The question to search for in the knowledge graph
        search_type: Type of search to perform ("local", "global", or "drift")
        
    Returns:
        str: The search response from GraphRAG
        
    Raises:
        ValueError: If an invalid search_type is provided
        Exception: If the search operation fails
        
    Example:
        >>> response = query_graph("What are the main revenue streams?")
        >>> print(response)
    """
    # We set this from the global as the LLM function call sometimes loses this arg
    search_type = _request.graph_query_type

    # Map search types to their corresponding async functions
    search_functions = {
        "local": _query_graph_async_local,
        "global": _query_graph_async_global,
        "drift": _query_graph_async_drift,
    }

    if search_type not in search_functions:
        search_type = "local"  # Default to local search if invalid type provided

    # Validate search type
    print(f"Using search type: {search_type}")
    search_func = search_functions[search_type]
    
    # Handle async execution in sync context with proper waiting
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            print("Found existing event loop, running in thread pool")
            
            # We're in an event loop, so we need to run in a thread to avoid blocking
            def run_async_in_new_loop():
                # Create a completely new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    print(f"Starting {search_type} search in new thread...")
                    result = new_loop.run_until_complete(search_func(question))
                    print(f"Completed {search_type} search in thread")
                    return result
                except Exception as e:
                    print(f"Error in thread execution: {e}")
                    raise
                finally:
                    new_loop.close()
            
            # Use ThreadPoolExecutor with proper timeout and error handling
            with ThreadPoolExecutor(max_workers=1) as executor:
                print("Submitting task to thread pool...")
                future = executor.submit(run_async_in_new_loop)
                
                # Wait for completion with a reasonable timeout (5 minutes)
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout
                    print("Thread pool task completed successfully")
                    return result
                except TimeoutError:
                    print("GraphRAG query timed out after 5 minutes")
                    raise Exception("GraphRAG query timed out")
                except Exception as e:
                    print(f"Thread pool execution failed: {e}")
                    raise
                    
        except RuntimeError as e:
            # No event loop running, we can use asyncio.run directly
            print("No existing event loop found, running directly")
            print(f"Starting {search_type} search...")
            result = asyncio.run(search_func(question))
            print(f"Completed {search_type} search")
            return result
            
    except Exception as e:
        print(f"Error in query_graph: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"GraphRAG query failed: {str(e)}")


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
    async_functions = AsyncFunctionTool(functions={query_graph})

    sync_toolset = ToolSet()
    sync_toolset.add(sync_functions)
    
    async_toolset = AsyncToolSet()
    async_toolset.add(async_functions)
    
    return sync_toolset, async_toolset

def _setup_agent_team_with_globals(question: str, search_query_type: str, graph_query_type: str, use_search: bool, use_graph: bool, use_web: bool, use_reasoning: bool, evaluation_mode: bool) -> str:
    """
    Set up and run the agent team using pre-loaded global resources.
    
    This function uses resources loaded at startup to avoid reloading heavy data.
    """
    global _graphrag_data, _language_models, _project_client

    if not _project_client:
        return "Error: Azure project client not initialized"
    
    if evaluation_mode:
        print("Running in evaluation mode, no WebSocket updates will be sent")
        EVALUATION_MODE = True
        WEBSOCKET_EVENTS_ENABLED = False
    
    # Use the shared agents client without 'with' statement to keep it open
    agents_client = _project_client.agents
    
    # Create tools and toolsets using pre-loaded data
    search_tool, bing_tool = _create_search_tools_with_type(_project_client, search_query_type)
    sync_toolset, async_toolset = _create_agent_toolsets()
    
    # Register all agent functions
    agents_client.enable_auto_function_calls({create_task, query_graph})
    
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
            RAG_AGENT_INSTRUCTIONS = config["RAG_AGENT_INSTRUCTIONS"].strip()
            KG_AGENT_INSTRUCTIONS = config["KG_AGENT_INSTRUCTIONS"].strip()
            BING_AGENT_INSTRUCTIONS = config["BING_AGENT_INSTRUCTIONS"].strip()
            RAG_AGENT_DESCRIPTION = config["RAG_AGENT_DESCRIPTION"].strip()
            KG_AGENT_DESCRIPTION = config["KG_AGENT_DESCRIPTION"].strip()
            BING_AGENT_DESCRIPTION = config["BING_AGENT_DESCRIPTION"].strip()

            if not use_reasoning:
                if use_search:
                    print(f"Using search type: {search_query_type}")
                    TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS += f"\n\n{RAG_AGENT_DESCRIPTION}"

                if use_graph:
                    print(f"Using graph type: {graph_query_type}")
                    TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS += f"\n\n{KG_AGENT_DESCRIPTION}"

                if use_web:
                    print(f"Using web type: {BING_AGENT_DESCRIPTION}")
                    TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS += f"\n\n{BING_AGENT_DESCRIPTION}"
            else:
                # Reasoning agent uses different instructions
                if use_search:
                    TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS += f"\n\n{RAG_AGENT_DESCRIPTION}"
                if use_graph:
                    TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS += f"\n\n{KG_AGENT_DESCRIPTION}"
                if use_web:
                    TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS += f"\n\n{BING_AGENT_DESCRIPTION}"
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
        if use_search:
            # RAG agent gets search tool
            search_toolset = ToolSet()
            search_toolset.add(search_tool)
            agent_team.add_agent(
                model=MODEL_DEPLOYMENT_NAME,
                name="RAG-agent-multi",
                instructions=(RAG_AGENT_INSTRUCTIONS),
                tools=search_tool.definitions,
                tool_resources=search_tool.resources,
                can_delegate=False,
            )
        if use_graph:
            agent_team.add_agent(
                model=MODEL_DEPLOYMENT_NAME,
                name="KG-agent-multi",
                instructions=(KG_AGENT_INSTRUCTIONS),
                can_delegate=False,
                tools=async_toolset.definitions,
                # Note: GraphRAG queries are handled via the registered functions
            )
        if use_web:
            # Bing agent gets bing tool
            bing_toolset = ToolSet()
            bing_toolset.add(bing_tool)
            agent_team.add_agent(
                model=MODEL_DEPLOYMENT_NAME,
                name="Bing-agent-multi",
                instructions=(BING_AGENT_INSTRUCTIONS),
                tools=bing_tool.definitions,
                tool_resources=bing_tool.resources,
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

def _create_search_tools_with_type(project_client: AIProjectClient, search_type: str) -> Tuple[AzureAISearchTool, BingCustomSearchTool]:
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
    bing_tool = BingCustomSearchTool(connection_id=bing_conn.id, instance_name=BING_CONFIGURATION)

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
        global _graphrag_data, _language_models, _project_client, _request
        _request = request

        if not all([_graphrag_data, _language_models, _project_client]):
            raise HTTPException(status_code=503, detail="Application not fully initialized")
        
        question = request.query
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Run the agent team with the question using pre-loaded resources
        markdown_response, context, thread_id, run_id, token_usage = _setup_agent_team_with_globals(question, request.search_query_type, request.graph_query_type, use_search=request.use_search,
                                                           use_graph=request.use_graph, use_web=request.use_web, use_reasoning=request.use_reasoning, evaluation_mode=request.evaluation_mode)
        
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
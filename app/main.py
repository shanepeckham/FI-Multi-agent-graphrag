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

from agent_team import AgentTeam, AgentTask
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
        ValueError: If required environment variables are missing
    """
    # Load environment variables from .env file
    env_file_path = os.getenv("ENV_FILE_PATH", "/Users/shanepeckham/sources/graphrag/app/.env")
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

# Sample questions for testing
SAMPLE_QUESTIONS = [
    "Are there any mine safety disclosures?",
    "What were the net sales by reportable segment for 2024, 2023 and 2022 for Japan?",
    "What is the percentage change for accessories net sales between 2023 and 2024?",
    "What would be the five-year cumulative total shareholder return if $100 was invested on September 2019 on the S&P 500 index?",
    "Given only the information provided to you, with no public record searches, evaluate the financial health of the company. What are the key indicators of financial health? What are the key risks to financial health? What are the key opportunities for financial health? What is your overall assessment of the company's financial health? Would you invest in this company? Why or why not?"
    "Are there any drops in revenue? If yes, what are the reasons for the drop? Which services/products are affected? What is the percentage drop in revenue? ",
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
def create_task(team_name: str, recipient: str, request: str, requestor: str) -> str:
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
        team = AgentTeam.get_team(team_name)
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
    return str(result.response)

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
    
    # Handle async execution in sync context
    try:
        loop = asyncio.get_running_loop()
        # We're in an event loop, so we need to run in a thread
        def run_in_thread():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(search_func(question))
            finally:
                new_loop.close()
        
        # Run in a separate thread
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
            
    except RuntimeError:
        # No event loop running, we can use asyncio.run directly
        return asyncio.run(search_func(question))


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

def _setup_agent_team_with_globals(question: str, search_query_type: str, graph_query_type: str, use_search: bool, use_graph: bool, use_web: bool, use_reasoning: bool) -> str:
    """
    Set up and run the agent team using pre-loaded global resources.
    
    This function uses resources loaded at startup to avoid reloading heavy data.
    """
    global _graphrag_data, _language_models, _project_client
    
    if not _project_client:
        return "Error: Azure project client not initialized"
    
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
        agent_team = AgentTeam("cr_team", agents_client=agents_client)

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
                can_delegate=False
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
                toolset=bing_toolset,
                can_delegate=False
            )
        
        # Assemble and run the team
        agent_team.assemble_team()
        return agent_team.process_request(request=question)
    
    return "Error: MODEL_DEPLOYMENT_NAME is not set"

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
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "GraphRAG API"}



@app.post("/query_team", response_model=QueryResponse)
def query_team_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Query the agent team for financial analysis.
    
    This endpoint uses pre-loaded resources from application startup
    to provide fast responses without reloading data. No authentication required.
    """
    try:
        # Use global variables loaded at startup
        global _graphrag_data, _language_models, _project_client
        
        if not all([_graphrag_data, _language_models, _project_client]):
            raise HTTPException(status_code=503, detail="Application not fully initialized")
        
        question = request.query
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Run the agent team with the question using pre-loaded resources
        markdown_response = _setup_agent_team_with_globals(question, request.search_query_type, request.graph_query_type, use_search=request.use_search, use_graph=request.use_graph, use_web=request.use_web, use_reasoning=request.use_reasoning)
        
        return QueryResponse(
            response=markdown_response,
            query=question
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
# Azure Services Quick Setup Guide for GraphRAG Multi-Agent Financial Analysis System

This quick start guide provides step-by-step instructions for setting up all the necessary Azure services required for the GraphRAG Multi-Agent Financial Analysis System.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Azure Resource Overview](#azure-resource-overview)
3. [Step 1: Azure AI Foundry (Formerly Azure AI Studio) Setup](#step-1-azure-ai-foundry-formerly-azure-ai-studio-setup)
4. [Step 2: Azure OpenAI Service Setup and Model Deployments](#step-2-azure-openai-service-setup-and-model-deployments)
5. [Step 3: Azure AI Search Service Setup](#step-3-azure-ai-search-service-setup)
6. [Step 4: Bing Search Service Setup](#step-4-bing-search-service-setup)
7. [Step 5: Application Insights Setup (Optional)](#step-5-application-insights-setup-optional)
8. [Next Steps](#next-steps)

## Prerequisites

Before starting, ensure you have:

- **Azure Subscription**: Active Azure subscription with sufficient credits/budget
- **Azure CLI**: [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- **Azure Portal Access**: Access to [Azure Portal](https://portal.azure.com)
- **Permissions**: Owner or Contributor access to the Azure subscription
- **Regional Considerations**: Choose a region that supports all required services (recommended: East US, West Europe, or West US 2)

### Required Service Quotas

Ensure your subscription has sufficient quotas for:
- Azure OpenAI Service (GPT-4 and embedding models)
- Azure AI Search (Standard tier recommended)
- Azure AI Foundry projects

## Azure Resource Overview

The system requires the following Azure services:

| Service | Purpose | 
|---------|---------|
| **Azure OpenAI Service** | LLM models (GPT-4.1, o3-mini, embeddings) |
| **Azure AI Foundry** | AI Agents and project management |
| **Azure AI Search** | Document indexing and search |
| **Bing Search API** | External web search capabilities |
| **Application Insights** | Monitoring and telemetry |

> **⚠️ Cost Warning**: These services can incur significant costs, especially Azure OpenAI. Monitor your usage closely and set up billing alerts.

## Step 1: Azure AI Foundry (Formerly Azure AI Studio) Setup

### 1.1 Create AI Foundry Project

1. **Access AI Foundry**
   - Go to [AI Foundry](https://ai.azure.com/)
   - Sign in with your Azure credentials

2. **Create New Project**
   - Click "Create Project"
   - **Project Name**: `graphrag-financial-agents` or other relevant name
   - **Subscription**: Select your subscription
   - **Resource Group**: Create new or use existing
   - **Region**: Choose a region that supports all required services (recommended: East US, West Europe, or West US 2)

3. **Configure Project Settings**
   - **AI Services**: This will create an Azure OpenAI resource automatically
   - **Storage**: Create new storage account
   - **Key Vault**: Create new Key Vault

### 1.2 Enable AI Agents

1. **Navigate to Agents Section**
   - In your AI Foundry project
   - Go to "Agents" in the left navigation

2. **Enable Agents Feature**
   - Follow prompts to enable AI Agents
   - This may require preview feature enablement

### 1.3 Get Project Details

1. **Copy Project Details**
   - In project overview, go to "Endpoints and keys"
   - Copy the **API Key**, **Azure AI Foundry project endpoint** and **Azure OpenAI endpoint**
   
   ```bash
   # You'll need these for environment variables:
   AZURE_OPENAI_API_KEY=your_key_here
   PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project-name
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```

## Step 2: Model Deployments

### 2.1 Deploy Required Models

Deploy the following models in your Azure OpenAI resource:

#### Model Deployments Required:

1. **Main Model**
   - **Deployment Name**: `gpt-4.1`
   - **Model**: `gpt-4.1`

2. **Reasoning Model**
   - **Deployment Name**: `o3-mini`
   - **Model**: `o3-mini`

3. **Text Embedding Model**
   - **Deployment Name**: `text-embedding-ada-002`
   - **Model**: `text-embedding-ada-002`

#### Deployment Steps:

1. **Access Model Management**
   - In your Azure AI foundry project go to "Models + endpoints"
   - Click "Deploy model" to deploy the necessary models

2. **Deploy Each Model**
   - Select the model you want to deploy
   - Set the deployment name as specified above
   - Choose appropriate capacity based on your needs and select the correct region
   - Click "Deploy"

You can also use Azure CLI to deploy models:
```bash
# Using Azure CLI to deploy models
az cognitiveservices account deployment create \
  --resource-group "your-rg" \
  --account-name "your-openai-resource" \
  --deployment-name "gpt-4.1" \
  --model-name "gpt-4" \
  --model-version "latest" \
  --sku-capacity 10 \
  --sku-name "Standard"
```

## Step 3: Azure AI Search Service Setup

### 3.1 Create Azure AI Search Resource

1. **Create Search Service**
   ```bash
   # Using Azure CLI
   az search service create \
     --name "your-search-service" \
     --resource-group "your-rg" \
     --location "eastus" \
     --sku "Standard" \
     --partition-count 1 \
     --replica-count 1
   ```

2. **Portal Configuration**
   - **Service Name**: `your-project-search` (globally unique)
   - **Subscription**: Your subscription
   - **Resource Group**: Same as other resources
   - **Location**: Same region
   - **Pricing Tier**: Standard (recommended for production)

### 3.2 Configure Search Index

1. **Create Index for Documents**
   - Index Name: `report_agent` (or customize via `AI_SEARCH_INDEX_NAME`)
   - This will be populated with your financial documents

2. **Note Search Service Details**
   ```bash
   # You'll need:
   AI_SEARCH_INDEX_NAME=report_agent
   # The connection will be configured in AI Foundry
   ```
3. **Import and index data**
   - Upload your documents to an Azure Blob Storage
   - Click "Import data" or "Import and vectorize data" from container with data
   - Create semantic configurations and vector profiles for the index

### 3.3 Create Connection in AI Foundry

1. **Navigate to Connections**
   - In your AI Foundry project go to "Management center"
   - Go to "Connected resources" section

2. **Add Azure AI Search Connection**
   - **Connection Name**: `agentsearcher`
   - **Service**: Select your Azure AI Search service
   - **Authentication**: API Key (recommended) or Managed Identity

3. **Add Azure Storage Account**
   - Add the storage account with the documents for the search service as a connected resource in the AI Foundry project

## Step 4: Bing Search Service Setup

### 4.1 Create Bing Search Resource

1. **Create Bing Search API**
   ```bash
   # Using Azure CLI
   az cognitiveservices account create \
     --name "your-bing-search" \
     --resource-group "your-rg" \
     --kind "Bing.Search.v7" \
     --sku "S1" \
     --location "global"
   ```

2. **Portal Configuration**
   - Search for "Bing Search" in Azure Portal
   - **Name**: `your-project-bing-search`
   - **Pricing Tier**: S1 (Standard)
   - **Location**: Global

### 4.2 Configure Bing Custom Search (Optional)

1. **Create Custom Search Instance**
   - Go to [Bing Custom Search](https://www.customsearch.ai/)
   - Create new instance for financial/business domains
   - **Configuration Name**: `sky` (default in project)

### 4.3 Create Connection in AI Foundry

1. **Add Bing Search Connection**
   - **Connection Name**: `agentbing`
   - **Service**: Select your Bing Search service
   - **Configuration**: If using custom search, specify configuration ID

## Step 5: Application Insights Setup (Optional)

### 5.1 Create Application Insights

1. **Create Resource**
   ```bash
   # Using Azure CLI
   az monitor app-insights component create \
     --app "your-app-insights" \
     --location "eastus" \
     --resource-group "your-rg" \
     --application-type "web"
   ```

2. **Get Connection String**
   - Navigate to your Application Insights resource
   - Copy the **Connection String** from the overview page

   ```bash
   APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=your-key;IngestionEndpoint=...
   ```

## Next Steps

After completing this setup:

1. **Document Indexing**: Prepare and index your financial documents using GraphRAG
2. **Application Deployment**: Deploy the application using Docker or Azure services
3. **Testing**: Run end-to-end tests with sample queries
4. **Monitoring**: Set up basic monitoring and alerting

## Additional Resources

- [Azure OpenAI Service Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure AI Foundry Documentation](https://docs.microsoft.com/en-us/azure/ai-studio/)
- [Azure AI Search Documentation](https://docs.microsoft.com/en-us/azure/search/)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)

---

**⚠️ Important Security Notes:**
- Never commit API keys or secrets to version control
- Use Azure Key Vault for production secrets management
- Regularly rotate API keys and access tokens
- Implement network security groups and private endpoints for production
- Enable audit logging for all services

**💡 Pro Tips:**
- Start with development pricing tiers and scale up
- Set up basic monitoring from day one
- Document your specific configuration for team members

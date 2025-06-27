# GraphRAG API Security Configuration

## API Key Authentication

The GraphRAG API now requires Bearer token authentication for secure access to the `/query_team` endpoint.

### Setup

1. **Configure API Key**: Set your API key in the `.env` file:
   ```
   API_KEY=your_secure_api_key_here
   ```

2. **Generated Key**: If no API key is provided, a secure key will be auto-generated. Check the application logs on startup.

3. **Use the API Key**: Include the API key in the Authorization header for all requests to `/query_team`:
   ```
   Authorization: Bearer your_api_key_here
   ```

### Example API Calls

#### Using curl:
```bash
curl -X POST "http://localhost:8000/query_team" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main revenue streams?",
    "graph_query_type": "local",
    "search_query_type": "SIMPLE",
    "use_search": true,
    "use_graph": true,
    "use_web": false,
    "use_reasoning": false
  }'
```

#### Using Python requests:
```python
import requests

headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}

data = {
    "query": "What are the main revenue streams?",
    "graph_query_type": "local",
    "search_query_type": "SIMPLE",
    "use_search": True,
    "use_graph": True,
    "use_web": False,
    "use_reasoning": False
}

response = requests.post("http://localhost:8000/query_team", headers=headers, json=data)
print(response.json())
```

### Security Features

- **Bearer Token Authentication**: All requests to `/query_team` require a valid API key
- **Secure Key Generation**: Uses cryptographically secure random key generation
- **Hash-based Verification**: API keys are hashed using SHA-256 for secure comparison
- **Timing Attack Protection**: Uses `secrets.compare_digest()` to prevent timing attacks
- **Error Handling**: Returns proper HTTP 401 Unauthorized responses for invalid/missing keys

### Endpoints

- `GET /` - Public endpoint (no authentication required)
- `GET /health` - Public health check (no authentication required) 
- `POST /query_team` - **Requires Bearer token authentication**

### Environment Variables

Add to your `.env` file:
```
# Required
AZURE_OPENAI_API_KEY=your_azure_openai_key

# API Security (optional - will be auto-generated if not provided)
API_KEY=your_secure_api_key

# Other configuration...
```

### Generated API Key Example

If you don't set an API key, one will be generated automatically:
```
Generated API Key: graphrag_Y01aHrEQQbU6JJZevbtkFmRzj3JNBqTqwwqy5D4X0dM
Use this key in the Authorization header as: Bearer graphrag_Y01aHrEQQbU6JJZevbtkFmRzj3JNBqTqwwqy5D4X0dM
```

### Testing Security

Try accessing the endpoint without authentication (should return 401):
```bash
curl -X POST "http://localhost:8000/query_team" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

Response:
```json
{
  "detail": "Missing API key"
}
```

# API Query UI

A Python GUI application for querying team data via API endpoints with beautiful markdown rendering.

## Features

- **Clean Interface**: Modern tkinter-based GUI with intuitive controls
- **API Integration**: Make POST requests to any API endpoint
- **Long-Running Request Support**: 
  - Configurable timeout (default 10 minutes)
  - Cancel functionality for ongoing requests
  - Real-time elapsed time display
  - Progress indicators
- **Markdown Rendering**: Custom markdown renderer that supports:
  - Headers (H1, H2, H3)
  - **Bold** and *italic* text
  - `Inline code` and code blocks
  - Bulleted and numbered lists
  - Links (visual styling)
- **Threaded Requests**: Non-blocking API calls with progress indicators
- **Error Handling**: User-friendly error messages and status updates
- **Configurable**: Easy to modify endpoint URLs, request parameters, and timeouts

## Requirements

- Python 3.7+ with tkinter support
- tkinter (usually included with Python, but may need special installation on some systems)
- requests library

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting tkinter on macOS

If you get `ModuleNotFoundError: No module named '_tkinter'`, you may need to:

1. **Use system Python**: Make sure you're using Python that has tkinter:
   ```bash
   python3 -c "import tkinter; print('tkinter works!')"
   ```

2. **Install Python with tkinter**: If using Homebrew:
   ```bash
   brew install python-tk
   ```

3. **Use the provided run script**: 
   ```bash
   ./run.sh
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
   
   Or use the provided run script (macOS/Linux):
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

2. Configure your API settings:
   - **API Endpoint**: Enter your API URL (default: `http://0.0.0.0:8000/query_team`)
   - **Bearer Token**: Optional authorization token for secure API access (masked input)
   - **Query**: Enter your question or query text
   - **Graph Query Type**: Select from `local`, `drift`, or `global`
   - **Search Query Type**: Select from `SIMPLE` or `SEMANTIC`
   - **Options**: Configure boolean settings:
     - **Use Search**: Enable/disable search functionality (default: enabled)
     - **Use Graph**: Enable/disable graph functionality (default: enabled)
     - **Use Web**: Enable/disable web functionality (default: disabled)
     - **Use Reasoning**: Enable/disable reasoning capabilities for complex queries (default: disabled)
   - **Timeout**: Set request timeout in minutes (default: 10 minutes)

3. Click "Submit Query" to send the request

4. **For long-running requests**:
   - Monitor the elapsed time in the progress area
   - Use the "Cancel" button to stop the request if needed
   - The timeout can be adjusted based on your expected response time

5. View the formatted response in the response area

## Testing

Run the test script to verify everything is working:
```bash
python test_app.py
```

## API Format

The application is designed to work with APIs that accept POST requests with this format:

```json
{
  "query": "What will the revenue in 2030 be?",
  "graph_query_type": "local",
  "search_query_type": "SIMPLE",
  "use_search": true,
  "use_graph": true,
  "use_web": false,
  "use_reasoning": false
}
```

### Request Headers:
- **Content-Type**: `application/json`
- **Accept**: `application/json`
- **Authorization**: `Bearer {token}` (optional, only sent if Bearer Token is provided in UI)

### Request Headers:
- **Content-Type**: `application/json`
- **Accept**: `application/json`

### Field Options:
- **graph_query_type**: `local`, `drift`, or `global`
- **search_query_type**: `SIMPLE` or `SEMANTIC`
- **use_search**: `true` or `false` (default: true)
- **use_graph**: `true` or `false` (default: true)
- **use_web**: `true` or `false` (default: false)
- **use_reasoning**: `true` or `false` (default: false)

And return markdown-formatted responses.

## Example Usage

The default configuration works with an endpoint that accepts queries like:
- "What will the revenue in 2030 be?"
- "Show me the sales trends"
- "Analyze customer feedback"

With configurable graph query types (local/drift/global), search query types (SIMPLE/SEMANTIC), and boolean options for search, graph, web, and reasoning functionality.

## Development

The application consists of several key components:

- `APIQueryUI`: Main application class handling the GUI
- `MarkdownRenderer`: Custom markdown rendering engine for tkinter
- Threading support for non-blocking API requests
- Comprehensive error handling and user feedback

## Customization

You can easily customize the application by:
- Modifying the default endpoint URL
- Adding new markdown formatting rules
- Extending the UI with additional input fields
- Adding authentication headers or other request parameters

## License

This project is open source and available under the MIT License.

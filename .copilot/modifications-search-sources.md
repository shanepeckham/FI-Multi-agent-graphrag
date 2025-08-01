# Search Sources Panel Implementation

## Summary
Added a new "Search Sources" panel beneath the existing "KG Sources" panel to display parsed search response data with real-time WebSocket updates.

## Changes Made

### 1. UI Dashboard Enhancement (`UI/realtime_dashboard.html`)
- **Added Search Sources Panel**: Created a new panel section beneath KG Sources with:
  - 🔍 Search Sources title
  - 📊 Latest Response section showing metadata counts
  - 📄 Source Texts section displaying parsed chunk data
- **Added JavaScript Event Handler**: Implemented `search_sources_update` event handling
- **Added Update Methods**: 
  - `updateSearchSources()`: Updates metadata counts and timestamps
  - `updateSearchSourceTexts()`: Displays chunk data with formatting

### 2. WebSocket Event Emission (`app/agent_team_dashboard.py`)
- **Added emit_search_sources_update() Function**: New function to emit search sources data via WebSocket
  - Sends chunk_id, parent_id, and chunk arrays
  - Includes timestamp and update_type metadata
  - Follows same pattern as existing KG sources emission
- **Integrated Search Sources Emission**: Added call to emit search sources when RAG-agent-multi completes processing
  - Triggers after `parse_search_sources()` function processes the response
  - Automatically updates dashboard in real-time

### 3. Data Flow
```
RAG-agent-multi response → parse_search_sources() → emit_search_sources_update() → WebSocket → Dashboard Update
```

## Features
- **Real-time Updates**: Search sources display immediately when RAG agent completes
- **Structured Display**: Shows chunk IDs, parent IDs, and formatted chunk content
- **Consistent UI**: Matches existing KG Sources panel styling and behavior
- **Error Handling**: Includes fallback display for empty or missing data

## Usage
The Search Sources panel will automatically populate when:
1. A user query is processed by the RAG-agent-multi agent
2. The agent response contains search source data in the expected format
3. The `parse_search_sources()` function successfully extracts the data
4. WebSocket events are available and functioning

## Data Format Expected
```json
{
  "chunk_id": ["2", "3"],
  "parent_id": ["1", "1"], 
  "chunk": ["(Inventories: $5,351 million...)", "(Revenue: $10,234 million...)"]
}
```

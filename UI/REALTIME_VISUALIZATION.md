# 🚀 Real-time Agent Team Visualization

This system provides **real-time visualization** of your GraphRAG Multi-Agent Team collaboration using WebSockets and a modern web dashboard.

![Realtime Dashboard](./media/Realtimedashboard.png)

## 🌟 Features

### 📊 **Real-time Dashboard**
- **Live Agent Monitoring**: See agent status (idle, working, completed)
- **Task Flow Visualization**: Track tasks as they move through the team
- **Activity Timeline**: Real-time log of all agent activities
- **Performance Metrics**: Live counters for active agents, completed tasks, etc.
- **Modern UI**: Beautiful, responsive interface with animations

### 🔄 **WebSocket Integration**
- **Instant Updates**: No refresh needed - see changes as they happen
- **Event-driven Architecture**: Captures all agent lifecycle events
- **Session Management**: Multiple users can monitor simultaneously
- **Auto-reconnection**: Handles network disconnections gracefully

### 🤖 **Agent Team Events**
- **Agent Creation**: When new agents are instantiated
- **Task Assignment**: When tasks are delegated to agents
- **Task Progress**: Real-time status updates during execution
- **Task Completion**: Results and outcomes from agent work
- **Team Coordination**: High-level team orchestration events

## 🚀 Quick Start

### 1. **Start Your Application**
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. **Open the Dashboard**
Navigate to: http://localhost:8000/dashboard

### 3. **Send API Requests**
The dashboard will automatically show real-time updates when you query the team:

```bash
curl -X POST "http://localhost:8000/query_team" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the revenue trends for Q4?",
    "graph_query_type": "global",
    "search_query_type": "SEMANTIC",
    "use_search": true,
    "use_graph": true,
    "use_web": false,
    "use_reasoning": false
  }'
```

### 4. **Run Test Script** (Optional)
```bash
python test_visualization.py
```

## 📋 Dashboard Sections

### 🔗 **Connection Status**
- Green indicator: Connected to WebSocket
- Red indicator: Disconnected (will auto-reconnect)
- Session ID and connection info

### 📊 **Team Metrics**
- **Active Agents**: Currently working agents
- **Completed Tasks**: Total tasks finished
- **Pending Tasks**: Tasks waiting to be processed

### 🤖 **Agent Cards**
Each agent appears as a card showing:
- **Agent Name**: (TeamLeader, RAG-agent, KG-agent, Bing-agent)
- **Status**: idle 💤, working ⚡, or completed ✅
- **Current Task**: What the agent is currently working on
- **Visual Indicators**: Color-coded borders and animations

### 📋 **Task List**
Real-time task tracking:
- **Task Description**: What needs to be done
- **Recipient**: Which agent is handling it
- **Status**: pending ⏸️, in-progress ⏳, completed ✅
- **Requestor**: Who assigned the task

### 📝 **Activity Log**
Chronological timeline of all events:
- **Timestamps**: Precise timing of each event
- **Event Types**: Color-coded categories (agent, task, team)
- **Details**: Descriptions and context for each activity
- **Auto-scroll**: Latest events appear at the top

## 🏗️ Technical Architecture

### **WebSocket Flow**
```
1. Dashboard connects to /ws/{session_id}
2. Agent team emits events during execution
3. WebSocket manager broadcasts to all connected clients
4. Dashboard updates UI in real-time
```

### **Event Types**
```javascript
// Agent Events
agent_created       // New agent instantiated
agent_started_task  // Agent begins working
agent_completed_task // Agent finishes work

// Task Events  
task_created        // New task assigned
task_started        // Task execution begins
task_completed      // Task finished with results

// Team Events
processing_started  // Team begins processing request
processing_completed // Team finishes processing
```

### **Component Files**
```
📁 app/
├── websocket_manager.py     # WebSocket connection management
├── websocket_events.py      # Event emission utilities
├── agent_team.py           # Enhanced with event emission
└── main.py                 # FastAPI with WebSocket endpoints

📁 UI/
└── realtime_dashboard.html # Real-time web dashboard
```

## 🎯 Use Cases

### **Development & Debugging**
- **Monitor agent behavior** during development
- **Debug task delegation** and see execution flow
- **Identify bottlenecks** in agent coordination
- **Validate task completion** and results

### **Production Monitoring**
- **Live system health** monitoring
- **Performance tracking** of agent team
- **User session monitoring** in real-time
- **System diagnostics** and troubleshooting

### **Demonstrations & Presentations**
- **Visual storytelling** of agent collaboration
- **Live demos** of multi-agent systems
- **Educational tools** for understanding AI workflows
- **Client presentations** showing system capabilities

## 🛠️ Customization

### **Adding New Event Types**
1. Add event emission in `agent_team.py`:
```python
event_emitter.emit_sync("custom_event", "agent", {
    "agent_name": agent_name,
    "custom_data": data
})
```

2. Handle in dashboard JavaScript:
```javascript
case 'agent_event':
    if (message.event_type === 'custom_event') {
        // Handle your custom event
    }
    break;
```

### **Styling the Dashboard**
Edit the CSS in `realtime_dashboard.html`:
- **Colors**: Update the CSS variables
- **Layout**: Modify the grid system
- **Animations**: Adjust keyframe animations
- **Icons**: Change emoji indicators

### **WebSocket Configuration**
Modify `websocket_manager.py`:
- **Connection limits**: Set max connections
- **Event filtering**: Filter events by type
- **Session management**: Add authentication
- **Logging**: Enhanced debug logging

## 📱 Mobile Support

The dashboard is **responsive** and works on:
- 📱 **Mobile phones** (portrait/landscape)
- 📱 **Tablets** (iPad, Android tablets)  
- 💻 **Desktop browsers** (Chrome, Firefox, Safari)
- 🖥️ **Large screens** (4K monitors)

## 🔧 Troubleshooting

### **Dashboard Not Loading**
- Check FastAPI is running on port 8000
- Verify `/dashboard` endpoint is accessible
- Check browser console for JavaScript errors

### **WebSocket Connection Failed**
- Ensure WebSocket endpoint `/ws/{session_id}` is available
- Check for firewall blocking WebSocket connections
- Verify browser supports WebSockets (all modern browsers do)

### **No Real-time Updates**
- Check agent_team.py has event emission code
- Verify websocket_events.py is imported correctly
- Check browser network tab for WebSocket messages

### **Events Not Appearing**
- Check event emission syntax in agent code
- Verify WebSocket manager is initialized
- Check browser console for WebSocket errors

## 🚀 Next Steps

### **Enhancements You Can Add**
1. **Authentication**: Add user login to the dashboard
2. **Historical Data**: Store and replay past sessions
3. **Metrics Dashboard**: Add charts and graphs
4. **Multi-team Support**: Monitor multiple agent teams
5. **Alert System**: Notifications for specific events
6. **Export Data**: Download session logs as JSON/CSV

### **Integration Options**
1. **Slack/Teams**: Send alerts to chat channels
2. **Grafana**: Create monitoring dashboards
3. **Prometheus**: Export metrics for monitoring
4. **Database**: Store events for analysis
5. **Email**: Send summary reports

## 📖 Learn More

- **FastAPI WebSockets**: https://fastapi.tiangolo.com/advanced/websockets/
- **GraphRAG Documentation**: https://microsoft.github.io/graphrag/
- **Azure AI Agents**: https://docs.microsoft.com/azure/ai/
- **Real-time Web Apps**: Modern WebSocket patterns

---

**Happy Monitoring!** 🎉 Your agent team collaboration is now fully visible in real-time!

"""
WebSocket Manager for Real-time Agent Team Visualization
"""
import json
import asyncio
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time agent team visualization"""
    
    def __init__(self):
        # Store active connections
        self.active_connections: Set[WebSocket] = set()
        # Store session data
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Initialize session
        self.sessions[session_id] = {
            "start_time": datetime.now().isoformat(),
            "agents": [],
            "tasks": [],
            "status": "connected"
        }
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up session
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "disconnected"
            del self.sessions[session_id]
        
        logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.discard(conn)
    
    async def send_agent_event(self, event_type: str, data: Dict[str, Any]):
        """Send agent-related events to all clients"""
        message = {
            "type": "agent_event",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    async def send_task_event(self, event_type: str, data: Dict[str, Any]):
        """Send task-related events to all clients"""
        message = {
            "type": "task_event", 
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    async def send_team_event(self, event_type: str, data: Dict[str, Any]):
        """Send team-related events to all clients"""
        message = {
            "type": "team_event",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    async def send_run_event(self, event_type: str, data: Dict[str, Any]):
        """Send run-related events to all clients"""
        message = {
            "type": "run",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    async def send_progress_update(self, session_id: str, progress_data: Dict[str, Any]):
        """Send progress updates for a specific session"""
        message = {
            "type": "progress",
            "session_id": session_id,
            "data": progress_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a session"""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return self.sessions.copy()


# Global instance
websocket_manager = WebSocketManager()

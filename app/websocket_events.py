"""
Enhanced Agent Team with Real-time WebSocket Event Emission
"""
import asyncio
import json
from typing import Optional, Dict, Any


class WebSocketEventEmitter:
    """Handles emission of real-time events to WebSocket clients"""
    
    def __init__(self):
        self.websocket_manager = None
        self._initialize_websocket_manager()
    
    def _initialize_websocket_manager(self):
        """Lazy initialization of websocket manager to avoid circular imports"""
        try:
            from websocket_manager import websocket_manager
            self.websocket_manager = websocket_manager
        except ImportError:
            print("WebSocket manager not available")
            self.websocket_manager = None
    
    async def emit_agent_event(self, event_type: str, data: Dict[str, Any]):
        """Emit agent-related events"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.send_agent_event(event_type, data)
            except Exception as e:
                print(f"Error emitting agent event: {e}")
    
    async def emit_task_event(self, event_type: str, data: Dict[str, Any]):
        """Emit task-related events"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.send_task_event(event_type, data)
            except Exception as e:
                print(f"Error emitting task event: {e}")
    
    async def emit_team_event(self, event_type: str, data: Dict[str, Any]):
        """Emit team-related events"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.send_team_event(event_type, data)
            except Exception as e:
                print(f"Error emitting team event: {e}")
    
    def emit_sync(self, event_type: str, category: str, data: Dict[str, Any]):
        """Synchronous wrapper for emitting events"""
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # If we have a running loop, schedule the coroutine
                if category == "agent":
                    loop.create_task(self.emit_agent_event(event_type, data))
                elif category == "task":
                    loop.create_task(self.emit_task_event(event_type, data))
                elif category == "team":
                    loop.create_task(self.emit_team_event(event_type, data))
            except RuntimeError:
                # No running event loop, try to create a new one
                try:
                    if category == "agent":
                        asyncio.run(self.emit_agent_event(event_type, data))
                    elif category == "task":
                        asyncio.run(self.emit_task_event(event_type, data))
                    elif category == "team":
                        asyncio.run(self.emit_team_event(event_type, data))
                except Exception as e:
                    print(f"Could not run async event in new loop: {e}")
        except Exception as e:
            print(f"Error in sync emit: {e}")


# Global event emitter instance
event_emitter = WebSocketEventEmitter()

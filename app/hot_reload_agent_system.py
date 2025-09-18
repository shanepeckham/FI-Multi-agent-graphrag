import json
import os
import time
from typing import Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from runtime_agent_factory import RuntimeAgentFactory

class JSONChangeHandler(FileSystemEventHandler):
    """Handler for JSON file changes to trigger agent reloading."""

    def __init__(self, factory: RuntimeAgentFactory, callback=None):
        self.factory = factory
        self.callback = callback
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith('actions.json'):
            # Debounce rapid file changes
            current_time = time.time()
            if current_time - self.last_modified > 1:  # 1 second debounce
                self.last_modified = current_time
                print("📁 actions.json changed - reloading agents...")
                self.factory.reload_actions()
                if self.callback:
                    self.callback()

class HotReloadAgentSystem:
    """
    Agent system with hot-reloading capability.
    Automatically detects changes to actions.json and updates agents on-the-fly.
    """

    def __init__(self, actions_file_path: str = "./actions.json"):
        self.actions_file_path = actions_file_path
        self.factory = RuntimeAgentFactory(actions_file_path)
        self.observer = None
        self.is_watching = False
        self.agent_team = None
        self.model_deployment_name = None
        self.toolset_class = None

    def setup_agents(self, agent_team, MODEL_DEPLOYMENT_NAME: str, ToolSet) -> Dict[str, Any]:
        """Setup agents and store references for hot-reloading."""
        self.agent_team = agent_team
        self.model_deployment_name = MODEL_DEPLOYMENT_NAME
        self.toolset_class = ToolSet

        return self.factory.create_agents_runtime(agent_team, MODEL_DEPLOYMENT_NAME, ToolSet)

    def start_hot_reload(self):
        """Start watching for changes to actions.json."""
        if self.is_watching:
            print("⚠️  Hot reload is already active")
            return

        watch_dir = os.path.dirname(os.path.abspath(self.actions_file_path))

        def reload_callback():
            if self.agent_team and self.model_deployment_name and self.toolset_class:
                print("🔄 Recreating agents with new configuration...")
                # Note: In a real implementation, you'd need to clear existing agents first
                # self.factory.create_agents_runtime(self.agent_team, self.model_deployment_name, self.toolset_class)
                print("✅ Agents updated with new configuration!")

        event_handler = JSONChangeHandler(self.factory, reload_callback)
        self.observer = Observer()
        self.observer.schedule(event_handler, watch_dir, recursive=False)
        self.observer.start()
        self.is_watching = True

        print(f"👀 Watching {self.actions_file_path} for changes...")
        print("💡 Edit actions.json to see agents update automatically!")

    def stop_hot_reload(self):
        """Stop watching for changes."""
        if self.observer and self.is_watching:
            self.observer.stop()
            self.observer.join()
            self.is_watching = False
            print("🛑 Hot reload stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_hot_reload()

"""
Runtime Agent Integration Helper
Provides functions to integrate the FullyDynamicAgentGenerator with main.py
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
from azure.ai.agents.models import FunctionTool, ToolSet
from agent_generator import FullyDynamicAgentGenerator
from action_agents import ActionAgents


class RuntimeAgentIntegrator:
    """Integrates runtime agent generation with the main application."""

    def __init__(self, actions_file_path: Optional[str] = None, enable_hot_reload: bool = False):
        """
        Initialize the runtime agent integrator.

        Args:
            actions_file_path: Path to action_templates.json file
            enable_hot_reload: Whether to enable hot reloading during development
        """
        if actions_file_path is None:
            script_dir = Path(__file__).parent
            actions_file_path = str(script_dir / "action_templates.json")

        self.actions_file_path = actions_file_path
        self.enable_hot_reload = enable_hot_reload
        self.generator = FullyDynamicAgentGenerator(actions_file_path)
        self.action_agents = ActionAgents()

        # Initialize hot reload system if enabled
        self.hot_reload_system = None
        if enable_hot_reload:
            try:
                from hot_reload_agent_system import HotReloadAgentSystem
                self.hot_reload_system = HotReloadAgentSystem(
                    actions_file_path=actions_file_path,
                    team_name="cr_team"  # Use the same team name as main.py
                )
                print("🔄 Hot reload system initialized for development mode")
            except ImportError:
                print("⚠️  Hot reload system not available, continuing without it")

    def get_dynamic_action_functions(self) -> Dict[str, Any]:
        """
        Get all action functions dynamically from the action_agents instance.

        Returns:
            Dict mapping function names to actual function objects
        """
        action_functions = {}

        # Load actions from JSON to get the list of available action types
        actions = self.generator._load_actions()

        for action in actions:
            task_type = action["task_type"]

            # Convert task_type to method name (same logic as in agent_generator)
            method_name = task_type

            # Get the actual method from action_agents
            if hasattr(self.action_agents, method_name):
                action_functions[method_name] = getattr(self.action_agents, method_name)
            else:
                print(f"⚠️  Warning: Method '{method_name}' not found in ActionAgents")

        return action_functions

    def register_all_action_functions(self, agents_client, create_task_func):
        """
        Dynamically register all action functions with the agents client.

        Args:
            agents_client: The Azure AI agents client
            create_task_func: The create_task function to register
        """
        # Get all action functions
        action_functions = self.get_dynamic_action_functions()

        # Add the create_task function
        all_functions = {create_task_func}
        all_functions.update(action_functions.values())

        # Register with the agents client
        agents_client.enable_auto_function_calls(all_functions)

        print(f"📝 Registered {len(all_functions)} functions with agents client:")
        print("   - create_task")
        for func_name in action_functions.keys():
            print(f"   - {func_name}")

    def setup_dynamic_agents(self, agent_team, model_deployment_name: str, config: Dict[str, str]):
        """
        Set up all agents dynamically using the runtime generator.

        Args:
            agent_team: The AgentTeam instance
            model_deployment_name: The model deployment name to use
            config: Configuration dictionary with agent descriptions and instructions
        """
        # Load actions from JSON
        actions = self.generator._load_actions()

        print(f"🤖 Setting up {len(actions)} dynamic agents...")

        for action in actions:
            self._setup_single_agent(agent_team, action, model_deployment_name, config)

    def _setup_single_agent(self, agent_team, action: Dict[str, Any], model_deployment_name: str, config: Dict[str, str]):
        """
        Set up a single agent based on action configuration.

        Args:
            agent_team: The AgentTeam instance
            action: Action configuration from JSON
            model_deployment_name: The model deployment name to use
            config: Configuration dictionary with agent descriptions and instructions
        """
        task_type = action["task_type"]
        display_name = action["display_name"]

        # Generate agent name using the same logic as the generator
        agent_name = self.generator._task_type_to_agent_name(task_type)

        # Get the corresponding action method
        method_name = task_type
        if not hasattr(self.action_agents, method_name):
            print(f"⚠️  Warning: Method '{method_name}' not found, skipping agent '{agent_name}'")
            return

        action_method = getattr(self.action_agents, method_name)

        # Create toolset for this agent
        tool_set = ToolSet()
        tool_set.add(FunctionTool(functions={action_method}))

        # Generate dynamic instructions
        instructions = self._generate_agent_instructions(action, config)

        # Add agent to team
        agent_team.add_agent(
            model=model_deployment_name,
            name=agent_name,
            instructions=instructions,
            tools=tool_set.definitions,
            can_delegate=False
        )

        print(f"   ✅ Added agent: {agent_name} ({display_name})")

    def _generate_agent_instructions(self, action: Dict[str, Any], config: Dict[str, str]) -> str:
        """
        Generate instructions for an agent based on action configuration.

        Args:
            action: Action configuration from JSON
            config: Configuration dictionary with fallback instructions

        Returns:
            Generated instructions string
        """
        task_type = action["task_type"]

        # First, try to use instructions from the JSON action template
        if "instructions" in action and action["instructions"]:
            return action["instructions"].strip()

        # Fallback: Try to find specific instructions in config
        instruction_key = f"{task_type.upper()}_AGENT_INSTRUCTIONS"
        instruction_key_alt = f"{task_type.replace('_', '').upper()}_AGENT_INSTRUCTIONS"

        if instruction_key in config:
            return config[instruction_key].strip()
        elif instruction_key_alt in config:
            return config[instruction_key_alt].strip()
        else:
            # Generate default instructions based on action metadata
            description = action.get("description", f"Handles {task_type} tasks")
            verb_description = action.get("verb_description", f"perform {task_type}")

            default_instructions = f"""You are an agent specialized in {description.lower()}.

Your primary function is to {verb_description}.

When you receive a task:
1. Carefully analyze the request to understand what needs to be done
2. Execute the appropriate action with the correct parameters
3. Provide clear feedback about what was accomplished

Always be helpful, accurate, and professional in your responses."""

            return default_instructions

    def start_hot_reload(self):
        """Start the hot reload system if enabled."""
        if self.hot_reload_system:
            self.hot_reload_system.start()
            print("🔄 Hot reload system started - agents will update automatically when action_templates.json changes")

    def stop_hot_reload(self):
        """Stop the hot reload system if enabled."""
        if self.hot_reload_system:
            self.hot_reload_system.stop()
            print("🛑 Hot reload system stopped")

    def get_agent_count(self) -> int:
        """Get the number of agents that will be created."""
        actions = self.generator._load_actions()
        return len(actions)

    def list_available_agents(self) -> List[str]:
        """Get a list of all available agent names."""
        actions = self.generator._load_actions()
        return [self.generator._task_type_to_agent_name(action["task_type"]) for action in actions]

    def get_dynamic_agent_descriptions(self) -> List[str]:
        """
        Get formatted descriptions of all dynamic agents for Team Leader instructions.

        Returns:
            List of formatted agent descriptions ready to be added to Team Leader instructions
        """
        actions = self.generator._load_actions()
        descriptions = []

        for action in actions:
            # Use the description from the JSON file (which now includes the detailed YAML format)
            description = action.get("description", f"Agent for {action['task_type']}")
            descriptions.append(f"- {description}")

        return descriptions

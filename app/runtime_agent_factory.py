import json
import inspect
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import os

class RuntimeActionAgents:
    """
    Dynamically generated action agents that are created at runtime from actions.json.
    Methods are generated on-the-fly and added to the class instance.
    """

    def __init__(self, actions_file_path: str = "./actions.json"):
        self.actions_file_path = actions_file_path
        self.actions_data = self._load_actions()
        self._generate_methods()

    def _load_actions(self) -> List[Dict[str, Any]]:
        """Load actions from JSON file."""
        with open(self.actions_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _generate_methods(self):
        """Generate and attach methods to this instance dynamically."""
        for action in self.actions_data:
            method = self._create_method(action)
            method_name = action["task_type"]
            setattr(self, method_name, method)

    def _create_method(self, action: Dict[str, Any]) -> Callable:
        """Create a method dynamically based on action configuration."""
        task_type = action["task_type"]
        parameters = action.get("parameters", {})
        return_template = action.get("return_message_template", f"Completed {task_type}.")

        def dynamic_method(*args, **kwargs):
            # Get parameter names and values
            param_names = list(parameters.keys())

            # Handle both positional and keyword arguments
            param_values = {}
            for i, arg in enumerate(args[1:]):  # Skip self
                if i < len(param_names):
                    param_values[param_names[i]] = arg
            param_values.update(kwargs)

            # Format return message using template
            try:
                if '{' in return_template and param_values:
                    result = return_template.format(**param_values)
                else:
                    result = return_template
            except KeyError:
                # Fallback if template has missing keys
                result = f"Completed {task_type} with parameters: {param_values}"

            return result

        # Set method name and documentation
        dynamic_method.__name__ = task_type
        dynamic_method.__doc__ = action.get("description", f"Performs {task_type}")

        return dynamic_method

    def save_to_file(self, content: str, filename: str = None):
        """Built-in file saving method."""
        if filename is None:
            filename = datetime.now().isoformat() + ".json"
        os.makedirs("response", exist_ok=True)
        file_path = os.path.join("response", filename)
        with open(file_path, "w") as file:
            file.write(content + "\n")
        return f"Content saved to {file_path}."

class RuntimeAgentFactory:
    """
    Factory that creates agents on-the-fly at runtime without any pre-generated files.
    """

    def __init__(self, actions_file_path: str = "./actions.json"):
        self.actions_file_path = actions_file_path
        self.actions_data = self._load_actions()
        self.action_agents = RuntimeActionAgents(actions_file_path)

    def _load_actions(self) -> List[Dict[str, Any]]:
        """Load actions from JSON file."""
        with open(self.actions_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _task_type_to_agent_name(self, task_type: str) -> str:
        """Convert task_type to agent name format."""
        words = task_type.split('_')
        camel_case = ''.join(word.capitalize() for word in words)
        return f"{camel_case}-agent-multi"

    def _generate_agent_instructions(self, action: Dict[str, Any]) -> str:
        """Generate agent instructions on the fly."""
        task_type = action["task_type"]
        display_name = action.get("display_name", task_type.replace('_', ' ').title())
        description = action.get("description", f"Performs {task_type.replace('_', ' ')}")
        verb_description = action.get("verb_description", f"perform {task_type.replace('_', ' ')}")
        parameters = action.get("parameters", {})

        agent_name = self._task_type_to_agent_name(task_type)

        if parameters:
            param_desc = ", ".join([f"{k} ({v})" for k, v in parameters.items()])
            return f"You are an agent named {agent_name}. Use the {task_type} method to {verb_description}. You must accept the following parameters: {param_desc}."
        else:
            return f"You are an agent named {agent_name}. Use the {task_type} method to {verb_description}. No parameters required."

    def _generate_classifier_instructions(self) -> str:
        """Generate classifier instructions with current action data."""
        action_types = [action["task_type"] for action in self.actions_data]
        action_parameters = {action["task_type"]: action.get("parameters", {}) for action in self.actions_data}

        instructions = """You are a classifier agent named Classifier-agent-multi. You are an expert in extracting actions from a given transcript.

**Available Actions:**
"""

        for action in self.actions_data:
            task_type = action["task_type"]
            parameters = action.get("parameters", {})

            if parameters:
                param_str = ", ".join([f"{k}: {v}" for k, v in parameters.items()])
                instructions += f"- {task_type} (Parameters: {param_str})\n"
            else:
                instructions += f"- {task_type}\n"

        instructions += """
**OUTPUT FORMAT REQUIREMENT:**
You MUST return your response as a valid JSON object that follows this base structure:
```json
{
    "task_type": "string",
    "parameters": {}
}
```

**Instructions:**
- Extract actions that match the available actions listed above
- For actions with parameters, include parameter values when extractable from the transcript
- Return valid JSON in the specified format
- Store output in timestamped file in "response" directory
"""
        return instructions

    def _generate_team_leader_instructions(self) -> str:
        """Generate team leader instructions with current agent data."""
        categories = set()
        agent_descriptions = []

        for action in self.actions_data:
            category = action.get("category", "general")
            categories.add(category)

            agent_name = self._task_type_to_agent_name(action["task_type"])
            description = action.get("description", f"Performs {action['task_type'].replace('_', ' ')}")
            parameters = action.get("parameters", {})

            if parameters:
                param_names = ", ".join(parameters.keys())
                agent_descriptions.append(f"- {agent_name}: {description} ({param_names})")
            else:
                agent_descriptions.append(f"- {agent_name}: {description}")

        instructions = f"""You are an agent named 'TeamLeader'. You are a leader of a team of agents.
You are responsible for utilizing specialized agents to complete tasks.

**Available Agent Categories:**
{chr(10).join([f'- {cat.replace("_", " ").title()} agents' for cat in categories])}

**Team Members:**
{chr(10).join(agent_descriptions)}

Use create_task function to assign work to the most suitable agent based on the request type.
"""
        return instructions

    def create_agents_runtime(self, agent_team, MODEL_DEPLOYMENT_NAME: str, ToolSet) -> Dict[str, Any]:
        """
        Create and configure all agents at runtime from actions.json.
        This happens on-the-fly without any pre-generated files.
        """
        print("🚀 Creating agents dynamically at runtime...")

        created_agents = {}

        # Create action agents dynamically
        for action in self.actions_data:
            task_type = action["task_type"]
            agent_name = self._task_type_to_agent_name(task_type)
            instructions = self._generate_agent_instructions(action)

            # Create toolset with dynamically generated method
            toolset = ToolSet()
            if hasattr(self.action_agents, task_type):
                method = getattr(self.action_agents, task_type)
                toolset.add(method)

            # Add agent to team
            agent_team.add_agent(
                model=MODEL_DEPLOYMENT_NAME,
                name=agent_name,
                instructions=instructions,
                can_delegate=False,
                tools=toolset,
            )

            created_agents[task_type] = {
                "agent_name": agent_name,
                "instructions": instructions,
                "action_config": action
            }

            print(f"✅ Created {agent_name} ({action.get('category', 'general')})")

        # Create classifier agent
        classifier_instructions = self._generate_classifier_instructions()
        classifier_toolset = ToolSet()

        agent_team.add_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="Large_Language_Model_Classifier-multi",
            instructions=classifier_instructions,
            can_delegate=False,
            tools=classifier_toolset,
        )
        created_agents["classifier"] = {
            "agent_name": "Large_Language_Model_Classifier-multi",
            "instructions": classifier_instructions
        }
        print("✅ Created Large_Language_Model_Classifier-multi")

        # Create team leader agent
        team_leader_instructions = self._generate_team_leader_instructions()

        agent_team.add_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="TeamLeader",
            instructions=team_leader_instructions,
            can_delegate=True,
        )
        created_agents["team_leader"] = {
            "agent_name": "TeamLeader",
            "instructions": team_leader_instructions
        }
        print("✅ Created TeamLeader")

        print(f"🎉 Successfully created {len(created_agents)} agents at runtime!")

        return {
            "created_agents": created_agents,
            "action_types": [action["task_type"] for action in self.actions_data],
            "categories": list(set([action.get("category", "general") for action in self.actions_data])),
            "total_agents": len(created_agents)
        }

    def reload_actions(self):
        """Reload actions from JSON file (useful for hot-reloading during development)."""
        print("🔄 Reloading actions from JSON...")
        self.actions_data = self._load_actions()
        self.action_agents = RuntimeActionAgents(self.actions_file_path)
        print("✅ Actions reloaded successfully!")

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get a summary of all available agents and their capabilities."""
        summary = {
            "total_actions": len(self.actions_data),
            "categories": {},
            "agents": []
        }

        for action in self.actions_data:
            category = action.get("category", "general")
            if category not in summary["categories"]:
                summary["categories"][category] = 0
            summary["categories"][category] += 1

            summary["agents"].append({
                "task_type": action["task_type"],
                "agent_name": self._task_type_to_agent_name(action["task_type"]),
                "display_name": action.get("display_name", action["task_type"].replace('_', ' ').title()),
                "description": action.get("description", ""),
                "category": category,
                "parameters": list(action.get("parameters", {}).keys())
            })

        return summary

# Convenience function for easy integration
def setup_runtime_agents(agent_team, MODEL_DEPLOYMENT_NAME: str, ToolSet, actions_file: str = "./actions.json") -> Dict[str, Any]:
    """
    One-line setup for runtime agent creation.

    Usage:
        result = setup_runtime_agents(agent_team, MODEL_DEPLOYMENT_NAME, ToolSet)
        print(f"Created {result['total_agents']} agents!")
    """
    factory = RuntimeAgentFactory(actions_file)
    return factory.create_agents_runtime(agent_team, MODEL_DEPLOYMENT_NAME, ToolSet)

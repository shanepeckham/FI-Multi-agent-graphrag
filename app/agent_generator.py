import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from jinja2 import Template

class FullyDynamicAgentGenerator:
    """
    Generates everything dynamically from action_templates.json with no hardcoded names or descriptions.
    """

    def __init__(self, actions_file_path: str = "./action_templates.json"):
        self.actions_file_path = actions_file_path
        self.actions_data = self._load_actions()

    def _load_actions(self) -> List[Dict[str, Any]]:
        """Load actions from JSON file."""
        with open(self.actions_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _task_type_to_agent_name(self, task_type: str) -> str:
        """Convert task_type to agent name format."""
        words = task_type.split('_')
        camel_case = ''.join(word.capitalize() for word in words)
        return f"{camel_case}-agent-multi"

    def _generate_parameter_signature(self, parameters: Dict[str, str]) -> str:
        """Generate method parameter signature."""
        if not parameters:
            return ""

        param_strings = []
        for param_name, param_type in parameters.items():
            if param_type == "list":
                param_strings.append(f"{param_name}: list[str]")
            elif param_type == "int":
                param_strings.append(f"{param_name}: int")
            else:  # default to str
                param_strings.append(f"{param_name}: str")

        return ", " + ", ".join(param_strings)

    def _generate_parameter_description(self, parameters: Dict[str, str]) -> str:
        """Generate parameter description for agent description."""
        if not parameters:
            return "No parameters required"

        param_descriptions = []
        for param_name, param_type in parameters.items():
            param_descriptions.append(f"{param_name} ({param_type})")

        return f"Parameters: {', '.join(param_descriptions)}"

    def generate_all_configurations(self) -> Dict[str, Any]:
        """Generate all configurations from actions.json."""
        configurations = {
            "agent_descriptions": {},
            "agent_instructions": {},
            "action_types": [],
            "action_parameters": {},
            "agent_configs": {},
            "categories": set()
        }

        for action in self.actions_data:
            task_type = action["task_type"]
            display_name = action.get("display_name", task_type.replace('_', ' ').title())
            description = action.get("description", f"Performs {task_type.replace('_', ' ')}")
            verb_description = action.get("verb_description", f"perform {task_type.replace('_', ' ')}")
            parameters = action.get("parameters", {})
            return_template = action.get("return_message_template", f"Completed {task_type}.")
            category = action.get("category", "general")

            agent_name = self._task_type_to_agent_name(task_type)
            param_desc = self._generate_parameter_description(parameters)

            # Store basic data
            configurations["action_types"].append(task_type)
            configurations["action_parameters"][task_type] = parameters
            configurations["categories"].add(category)

            # Generate agent description
            desc_key = f"{task_type.upper()}_AGENT_DESCRIPTION"
            desc_value = f"- {agent_name}: {description}. {param_desc}."
            configurations["agent_descriptions"][desc_key] = desc_value

            # Generate agent instructions
            inst_key = f"{task_type.upper()}_AGENT_INSTRUCTIONS"
            if parameters:
                inst_value = f"You are an agent named {agent_name}. Use the {task_type} method to {verb_description}. You must accept the following parameters: {param_desc}."
            else:
                inst_value = f"You are an agent named {agent_name}. Use the {task_type} method to {verb_description}. No parameters required."
            configurations["agent_instructions"][inst_key] = inst_value

            # Store agent config
            configurations["agent_configs"][task_type] = {
                "agent_name": agent_name,
                "display_name": display_name,
                "description": description,
                "verb_description": verb_description,
                "method_name": task_type,
                "parameters": parameters,
                "return_template": return_template,
                "category": category,
                "param_signature": self._generate_parameter_signature(parameters)
            }

        configurations["categories"] = list(configurations["categories"])
        return configurations

    def generate_action_agents_class(self) -> str:
        """Generate the complete ActionAgents class from JSON data."""
        configs = self.generate_all_configurations()

        class_template = Template("""import os
from datetime import datetime

class ActionAgents:
    \"\"\"
    Action agent methods generated dynamically from actions.json.
    All method names, parameters, and return messages are defined in the JSON configuration.
    \"\"\"
{% for task_type, config in agent_configs.items() %}
    def {{ config.method_name }}(self{{ config.param_signature }}):
        # {{ config.description }}
        {% if config.parameters -%}
        {% for param_name in config.parameters.keys() -%}
        # Parameter: {{ param_name }} ({{ config.parameters[param_name] }})
        {% endfor -%}
        {% endif -%}
        {% if config.return_template and '{' in config.return_template -%}
        return "{{ config.return_template }}".format({{ config.parameters.keys() | join(', ') if config.parameters else '' }})
        {% else -%}
        return "{{ config.return_template }}"
        {% endif %}
{% endfor %}
    def _save_to_file_internal(self, content: str, filename: str = None):
        \"\"\"Internal file saving utility.\"\"\"
        if filename is None:
            filename = datetime.now().isoformat() + ".json"
        os.makedirs("response", exist_ok=True)
        file_path = os.path.join("response", filename)
        with open(file_path, "w") as file:
            file.write(content + "\\n")
        return file_path
""")

        return class_template.render(agent_configs=configs["agent_configs"])

    def generate_yaml_config(self) -> str:
        """Generate complete YAML configuration from JSON data."""
        configs = self.generate_all_configurations()

        yaml_template = Template("""# Fully dynamic configuration generated from actions.json
TEAM_LEADER_MODEL: |
    gpt-4o-mini

# Dynamically generated agent descriptions
{% for desc_key, desc_value in agent_descriptions.items() -%}
{{ desc_key }}: |
    {{ desc_value }}

{% endfor %}
# Dynamically generated agent instructions
{% for inst_key, inst_value in agent_instructions.items() -%}
{{ inst_key }}: |
    {{ inst_value }}

{% endfor %}
# Dynamic classifier configuration
CLASSIFIER_AGENT_INSTRUCTIONS: |
    You are a classifier agent named Classifier-agent-multi. You are an expert in extracting actions from a given transcript.

    **Available Actions:**
    {% for action_type in action_types -%}
    - {{ action_type }}{% if action_parameters[action_type] %} (Parameters: {% for param, param_type in action_parameters[action_type].items() %}{{ param }}: {{ param_type }}{% if not loop.last %}, {% endif %}{% endfor %}){% endif %}
    {% endfor %}

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

# Team leader instructions with dynamic agent awareness
TEAM_LEADER_INSTRUCTIONS_ALL_AGENTS: |
    You are an agent named 'TeamLeader'. You are a leader of a team of agents.
    You are responsible for utilizing specialized agents to complete tasks.

    **Available Agent Categories:**
    {% for category in categories -%}
    - {{ category.replace('_', ' ').title() }} agents
    {% endfor %}

    **Team Members:**
    {% for task_type, config in agent_configs.items() -%}
    - {{ config.agent_name }}: {{ config.description }}{% if config.parameters %} ({{ config.parameters.keys() | join(', ') }}){% endif %}
    {% endfor %}

    Use create_task function to assign work to the most suitable agent based on the request type.

TEAM_LEADER_INSTRUCTIONS_REASONING_ALL_AGENTS: |
    You are an agent named 'TeamLeader'. You are a banking assistant and leader of a team of agents.
    You are responsible for utilizing specialized agents to complete banking and KYC tasks.

    **Available Agent Categories:**
    {% for category in categories -%}
    - {{ category.replace('_', ' ').title() }} agents
    {% endfor %}

    **Team Members:**
    {% for task_type, config in agent_configs.items() -%}
    - {{ config.agent_name }}: {{ config.description }}{% if config.parameters %} ({{ config.parameters.keys() | join(', ') }}){% endif %}
    {% endfor %}

    Save classifier output for evaluation purposes.
    Use create_task function to assign work to the most suitable agent.
""")

        return yaml_template.render(**configs)

    def generate_main_integration_code(self) -> str:
        """Generate integration code for main.py."""
        configs = self.generate_all_configurations()

        integration_template = Template("""
# Fully dynamic agent setup - no hardcoded names or descriptions
def setup_fully_dynamic_agents(agent_team, action_agents, MODEL_DEPLOYMENT_NAME, ToolSet):
    \"\"\"Set up all agents dynamically from actions.json configuration.\"\"\"

    from fully_dynamic_agent_generator import FullyDynamicAgentGenerator

    generator = FullyDynamicAgentGenerator()
    configs = generator.generate_all_configurations()

    # Set up each agent dynamically
    {% for task_type, config in agent_configs.items() -%}
    # {{ config.display_name }} Agent ({{ config.category }})
    {{ task_type }}_toolset = ToolSet()
    if hasattr(action_agents, '{{ config.method_name }}'):
        {{ task_type }}_toolset.add(getattr(action_agents, '{{ config.method_name }}'))

    agent_team.add_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="{{ config.agent_name }}",
        instructions=configs["agent_instructions"]["{{ task_type.upper() }}_AGENT_INSTRUCTIONS"],
        can_delegate=False,
        tools={{ task_type }}_toolset,
    )

    {% endfor %}
    # Classifier agent with dynamic action awareness
    classifier_toolset = ToolSet()

    # Render classifier instructions with actual action data
    from jinja2 import Template
    classifier_template = Template(configs["classifier_instructions"])
    rendered_classifier_instructions = classifier_template.render(
        action_types=configs["action_types"],
        action_parameters=configs["action_parameters"]
    )

    agent_team.add_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="Large_Language_Model_Classifier-multi",
        instructions=rendered_classifier_instructions,
        can_delegate=False,
        tools=classifier_toolset,
    )

    return configs

# Usage in your main function:
# configs = setup_fully_dynamic_agents(agent_team, action_agents, MODEL_DEPLOYMENT_NAME, ToolSet)
""")

        return integration_template.render(agent_configs=configs["agent_configs"])

def regenerate_all_from_json():
    """Utility function to regenerate all files from actions.json."""
    generator = FullyDynamicAgentGenerator()

    # Generate ActionAgents class
    action_agents_code = generator.generate_action_agents_class()
    with open("action_agents_generated.py", "w") as f:
        f.write(action_agents_code)

    # Generate YAML config
    yaml_config = generator.generate_yaml_config()
    with open("agent_team_config_generated.yaml", "w") as f:
        f.write(yaml_config)

    # Generate integration code
    integration_code = generator.generate_main_integration_code()
    with open("main_integration_generated.py", "w") as f:
        f.write(integration_code)

    print("All files regenerated from actions.json!")

    return generator.generate_all_configurations()

if __name__ == "__main__":
    regenerate_all_from_json()

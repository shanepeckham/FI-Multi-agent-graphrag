# MADCAP Dynamic Agent System Summary

## Overview
The codebase has been significantly refactored to implement a fully dynamic multi-agent system that generates and configures agents at runtime using the `action_templates.json` file. This eliminates the need for pre-generated agent code and allows for flexible, configuration-driven agent creation.

## Key Architecture Changes

### 1. Dynamic Agent Configuration (`action_templates.json`)
The system now uses a JSON configuration file that defines agent behaviors, parameters, and instructions:

```json
{
  "task_type": "schedule_meeting",
  "display_name": "Schedule Meeting", 
  "description": "ScheduleMeeting-agent-multi: Schedules a meeting for a specified date and time.",
  "verb_description": "schedule a meeting",
  "parameters": {"date": "str", "time": "str"},
  "return_message_template": "Scheduled meeting on {date} at {time}.",
  "category": "scheduling",
  "instructions": "You are an agent named ScheduleMeeting-agent-multi..."
}
```

**Supported Categories:**
- Scheduling (`schedule_meeting`)
- KYC Management (`update_kyc_total_assets`, `update_kyc_origin_of_assets`, `update_kyc_purpose_of_businessrelation`, `update_kyc_activity`)
- Contact Management (`plan_contact`, `update_contact_info_non_postal`, `update_contact_info_postal_address`)
- Utility (`save_to_file`)

### 2. Runtime Agent Factory (`runtime_agent_factory.py`)

#### RuntimeActionAgents Class
- **Dynamic Method Generation**: Creates agent methods on-the-fly from JSON configuration
- **Parameter Handling**: Supports both positional and keyword arguments
- **Template-based Responses**: Uses configurable return message templates
- **Runtime Adaptation**: Can be updated without code changes

#### RuntimeAgentFactory Class
- **Agent Name Convention**: Converts `task_type` to `{CamelCase}-agent-multi` format
- **Instruction Generation**: Creates specialized agent instructions based on configuration
- **Classifier Integration**: Generates classifier instructions with available actions
- **Team Leader Coordination**: Creates team leader instructions with agent summaries

#### Key Methods:
```python
def create_agents_runtime(self, agent_team, MODEL_DEPLOYMENT_NAME, ToolSet):
    """Creates and configures all agents at runtime from actions.json"""
    
def reload_actions(self):
    """Reloads actions from JSON file for hot-reloading"""
    
def get_agent_summary(self):
    """Returns summary of all available agents and capabilities"""
```

### 3. Hot Reload System (`hot_reload_agent_system.py`)

#### Features:
- **File Watching**: Monitors `action_templates.json` for changes
- **Automatic Reloading**: Updates agents when configuration changes
- **Debouncing**: Prevents rapid consecutive reloads
- **Development Mode**: Can be enabled/disabled for production vs development

#### Usage:
```python
hot_reload_system = HotReloadAgentSystem("./action_templates.json")
hot_reload_system.start_hot_reload()
# Agents automatically update when JSON file changes
```

### 4. Integration Layer (`runtime_integration_helper.py`)

#### RuntimeAgentIntegrator Class:
- **Bridge Component**: Connects dynamic generation with existing `main.py` architecture
- **Function Registration**: Automatically registers action functions with Azure AI agents client
- **Hot Reload Support**: Optional integration with hot reload system
- **Configuration Management**: Handles paths and setup parameters

#### Key Integration Points:
```python
# In main.py
runtime_integrator = RuntimeAgentIntegrator(enable_hot_reload=False)
runtime_integrator.register_all_action_functions(agents_client, create_task)
runtime_integrator.setup_dynamic_agents(agent_team, MODEL_DEPLOYMENT_NAME, config)
```

### 5. Core Agent Implementation (`action_agents.py`)
Contains concrete implementations of all action methods that correspond to the `task_type` values in `action_templates.json`:

- **Schedule Meeting**: `schedule_meeting(date, time)`
- **KYC Operations**: Multiple methods for KYC data updates
- **Contact Management**: Methods for contact planning and updates
- **File Operations**: `save_to_file(content)` with timestamped output

## System Flow

### 1. Initialization Phase
```
main.py → RuntimeAgentIntegrator → FullyDynamicAgentGenerator → action_templates.json
```

### 2. Agent Creation
1. **Load Configuration**: Parse `action_templates.json`
2. **Generate Instructions**: Create agent-specific instructions
3. **Create Methods**: Generate dynamic methods with parameter validation
4. **Register Tools**: Add methods to agent toolsets
5. **Add to Team**: Configure agents with Azure AI Agents

### 3. Runtime Execution
1. **Request Processing**: Team leader receives user request
2. **Agent Selection**: Chooses appropriate specialist agent based on task type
3. **Method Execution**: Calls dynamically generated method
4. **Response Formatting**: Uses configured response templates

### 4. Hot Reload (Development Mode)
1. **File Monitoring**: Watch `action_templates.json` for changes
2. **Change Detection**: Trigger reload on file modification
3. **Agent Recreation**: Regenerate agents with new configuration
4. **Seamless Update**: Continue operation with updated agents

## Benefits of Dynamic Architecture

### 1. Configuration-Driven Development
- **No Code Generation**: Agents created purely from JSON configuration
- **Rapid Prototyping**: Add new agents by editing JSON file
- **Version Control**: Agent definitions tracked in source control

### 2. Maintainability
- **Single Source of Truth**: All agent definitions in one file
- **Consistent Patterns**: Standardized agent creation process
- **Easy Updates**: Modify behavior without touching code

### 3. Scalability
- **Unlimited Agents**: Add as many agents as needed
- **Category Organization**: Group related agents logically
- **Parameter Flexibility**: Support various parameter types and patterns

### 4. Development Experience
- **Hot Reload**: Instant feedback during development
- **Type Safety**: Parameter validation and type checking
- **Debugging**: Clear tracing and logging throughout

## Implementation Details

### Agent Naming Convention
```
task_type: "schedule_meeting" → Agent Name: "ScheduleMeeting-agent-multi"
task_type: "update_kyc_total_assets" → Agent Name: "UpdateKycTotalAssets-agent-multi"
```

### Parameter Handling
- **Type Specification**: Parameters include type information (`str`, `int`, `list`)
- **Validation**: Runtime validation of parameter types and presence
- **Flexibility**: Support for both positional and keyword arguments

### Response Templates
- **Parameterized**: Use `{parameter_name}` placeholders
- **Fallback**: Graceful handling of template errors
- **Consistency**: Standardized response format across agents

### Error Handling
- **Graceful Degradation**: Continue operation even with configuration errors
- **Detailed Logging**: Comprehensive error reporting and debugging information
- **Recovery**: Ability to reload configuration and recover from errors

## Usage Examples

### Adding a New Agent
1. **Edit JSON**: Add new entry to `action_templates.json`
2. **Implement Method**: Add corresponding method to `action_agents.py`
3. **Test**: Agent automatically available after reload

### Hot Reload Development
```python
# Enable hot reload in main.py
runtime_integrator = RuntimeAgentIntegrator(enable_hot_reload=True)

# Edit action_templates.json → Agents automatically update
```

### Production Deployment
```python
# Disable hot reload for production
runtime_integrator = RuntimeAgentIntegrator(enable_hot_reload=False)
```

This dynamic architecture transforms the agent system from static, code-generated agents to a flexible, configuration-driven system that can adapt to changing requirements without code modifications.

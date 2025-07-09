# pylint: disable=line-too-long,useless-suppression
# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import os
import yaml  # type: ignore

from opentelemetry import trace
from opentelemetry.trace import Span  # noqa: F401 # pylint: disable=unused-import
from typing import Any, Dict, Optional, Set, List
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ToolSet, Tool, ToolResources, MessageRole, Agent, AgentThread, AzureAISearchQueryType, AzureAISearchTool, ListSortOrder, MessageRole

# Import WebSocket event emitter conditionally
try:
    from websocket_events import event_emitter
    WEBSOCKET_EVENTS_AVAILABLE = True
except ImportError:
    print("WebSocket events not available")
    event_emitter = None
    WEBSOCKET_EVENTS_AVAILABLE = False

tracer = trace.get_tracer(__name__)


class _AgentTeamMember:
    """
    Represents an individual agent on a team.

    :param model: The model (e.g. GPT-4) used by this agent.
    :param name: The agent's name.
    :param instructions: The agent's initial instructions or "personality".
    :param toolset: An optional ToolSet with specialized tools for this agent.
    :param can_delegate: Whether this agent has delegation capability (e.g., 'create_task').
                         Defaults to True.
    """

    def __init__(
        self, model: str, name: str, instructions: str, toolset: Optional[ToolSet] = None, tool_resources: Optional[ToolResources] = None,
            tools: Optional[Tool] = None, can_delegate: bool = True
    ) -> None:
        self.tool_resources: Optional[ToolResources] = tool_resources
        self.tools: Optional[Tool] = tools
        self.model = model
        self.name = name
        self.instructions = instructions
        self.agent_instance: Optional[Agent] = None
        self.toolset: Optional[ToolSet] = toolset
        self.can_delegate = can_delegate


class AgentTask:
    """
    Encapsulates a task for an agent to perform.

    :param recipient: The name of the agent who should receive the task.
    :param task_description: The description of the work to be done or question to be answered.
    :param requestor: The name of the agent or user requesting the task.
    """

    def __init__(self, recipient: str, task_description: str, requestor: str) -> None:
        self.recipient = recipient
        self.task_description = task_description
        self.requestor = requestor


class AgentTeam:
    """
    A class that represents a team of agents.

    """

    # Static container to store all instances of AgentTeam
    _teams: Dict[str, "AgentTeam"] = {}

    _agents_client: AgentsClient
    _agent_thread: Optional[AgentThread] = None
    _team_leader: Optional[_AgentTeamMember] = None
    _members: List[_AgentTeamMember] = []
    _tasks: List[AgentTask] = []
    _team_name: str = ""
    _current_request_span: Optional[Span] = None
    _current_task_span: Optional[Span] = None

    def __init__(self, team_name: str, agents_client: AgentsClient):
        """
        Initialize a new AgentTeam and set it as the singleton instance.
        """
        # Validate that the team_name is a non-empty string
        if not isinstance(team_name, str) or not team_name:
            raise ValueError("Team name must be a non-empty string.")
        # Check for existing team with the same name
        #if team_name in AgentTeam._teams:
        #    raise ValueError(f"A team with the name '{team_name}' already exists.")
        self.team_name = team_name
        if agents_client is None:
            raise ValueError("No AgentsClient provided.")
        self._agents_client = agents_client
        # Store the instance in the static container
        AgentTeam._teams[team_name] = self

        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = current_dir.replace("utils", "app")
        # Construct the full path to the config file
        file_path = os.path.join(app_dir, "agent_team_config.yaml")
        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            self.TEAM_LEADER_INSTRUCTIONS = config["TEAM_LEADER_INSTRUCTIONS"]
            self.TEAM_LEADER_INITIAL_REQUEST = config["TEAM_LEADER_INITIAL_REQUEST"]
            self.TEAM_LEADER_TASK_COMPLETENESS_CHECK_INSTRUCTIONS = config[
                "TEAM_LEADER_TASK_COMPLETENESS_CHECK_INSTRUCTIONS"
            ]
            self.TEAM_MEMBER_CAN_DELEGATE_INSTRUCTIONS = config["TEAM_MEMBER_CAN_DELEGATE_INSTRUCTIONS"]
            self.TEAM_MEMBER_NO_DELEGATE_INSTRUCTIONS = config["TEAM_MEMBER_NO_DELEGATE_INSTRUCTIONS"]
            self.TEAM_LEADER_MODEL = config["TEAM_LEADER_MODEL"].strip()

    @staticmethod
    def get_team(team_name: str) -> "AgentTeam":
        """Static method to fetch the AgentTeam instance by name."""
        team = AgentTeam._teams.get(team_name)
        if team is None:
            raise ValueError(f"No team found with the name '{team_name}'.")
        return team

    @staticmethod
    def _remove_team(team_name: str) -> None:
        """Static method to remove an AgentTeam instance by name."""
        if team_name not in AgentTeam._teams:
            raise ValueError(f"No team found with the name '{team_name}'.")
        del AgentTeam._teams[team_name]

    def add_agent(
        self, model: str, name: str, instructions: str, toolset: Optional[ToolSet] = None,
            tools: Optional[Tool] = None, tool_resources: Optional[ToolResources] = None, can_delegate: bool = True
    ) -> None:
        """
        Add a new agent (team member) to this AgentTeam.

        :param tools:
        :param tool_resources:
        :param model: The model name (e.g. GPT-4) for the agent.
        :param name: The name of the agent being added.
        :param instructions: The initial instructions/personality for the agent.
        :param toolset: An optional ToolSet to configure specific tools (functions, etc.)
                        for this agent. If None, we'll create a default set.
        :param can_delegate: If True, the agent can delegate tasks (via create_task).
                            If False, the agent does not get 'create_task' in its ToolSet
                            and won't mention delegation in instructions.
        """
        if tools is None and tool_resources is None:
            toolset = ToolSet()

        if can_delegate:
            # If agent can delegate, ensure it has 'create_task'
            try:
                function_tool = toolset.get_tool(FunctionTool)
                function_tool.add_functions(agent_team_default_functions)
            except ValueError:
                default_function_tool = FunctionTool(agent_team_default_functions)
                toolset.add(default_function_tool)

        member = _AgentTeamMember(
            model=model,
            name=name,
            instructions=instructions,
            toolset=toolset,
            tools=tools,
            tool_resources=tool_resources,
            can_delegate=can_delegate,
        )
        self._members.append(member)

    def set_team_leader(self, model: str, name: str, instructions: str, toolset: Optional[ToolSet] = None) -> None:
        """
        Set the team leader for this AgentTeam.

        If team leader has not been set prior to the call to assemble_team,
        then a default team leader will be set.

        :param model: The model name (e.g. GPT-4) for the agent.
        :param name: The name of the team leader.
        :param instructions: The instructions for the team leader. These instructions
                             are not modified by the implementation, so all required
                             information about other team members and how to pass tasks
                             to them should be included.
        :param toolset: An optional ToolSet to configure specific tools (functions, etc.)
                        for the team leader.
        """
        member = _AgentTeamMember(model=model, name=name, instructions=instructions, toolset=toolset)
        self._team_leader = member

    def add_task(self, task: AgentTask) -> None:
        """
        Add a new task to the team's task list.

        :param task: The task to be added.
        """
        self._tasks.append(task)

    def _create_team_leader(self) -> None:
        """
        Create the team leader agent.
        """
        assert self._agents_client is not None, "agents_client must not be None"
        assert self._team_leader is not None, "team leader has not been added"

        self._team_leader.agent_instance = self._agents_client.create_agent(
            model=self._team_leader.model,
            name=self._team_leader.name,
            instructions=self._team_leader.instructions,
            toolset=self._team_leader.toolset,
        )
        
        # Emit agent created event
        if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
            event_emitter.emit_sync("agent_created", "agent", {
                "agent_name": self._team_leader.name,
                "model": self._team_leader.model,
                "instructions": self._team_leader.instructions[:200] + "..." if len(self._team_leader.instructions) > 200 else self._team_leader.instructions,
                "agent_type": "team_leader"
            })

    def _set_default_team_leader(self):
        """
        Set the default 'TeamLeader' agent with awareness of all other agents.
        """
        toolset = ToolSet()
        toolset.add(default_function_tool)
        instructions = self.TEAM_LEADER_INSTRUCTIONS.format(agent_name="TeamLeader", team_name=self.team_name) + "\n"
        # List all agents (will be empty at this moment if you haven't added any, or you can append after they're added)
        for member in self._members:
            instructions += f"- {member.name}: {member.instructions}\n"

        self._team_leader = _AgentTeamMember(
            model=self.TEAM_LEADER_MODEL,
            name="TeamLeader",
            instructions=instructions,
            toolset=toolset,
            can_delegate=True,
        )

    def assemble_team(self):
        """
        Create the team leader agent and initialize all member agents with
        their configured or default toolsets.
        """
        assert self._agents_client is not None, "agents_client must not be None"

        if self._team_leader is None:
            self._set_default_team_leader()

        self._create_team_leader()

        for member in self._members:
            if member is self._team_leader:
                continue

            team_description = ""
            for other_member in self._members:
                if other_member != member:
                    team_description += f"- {other_member.name}: {other_member.instructions}\n"

            if member.can_delegate:
                extended_instructions = self.TEAM_MEMBER_CAN_DELEGATE_INSTRUCTIONS.format(
                    name=member.name,
                    team_name=self._team_name,
                    original_instructions=member.instructions,
                    team_description=team_description,
                )
            else:
                extended_instructions = self.TEAM_MEMBER_NO_DELEGATE_INSTRUCTIONS.format(
                    name=member.name,
                    team_name=self._team_name,
                    original_instructions=member.instructions,
                    team_description=team_description,
                )
            member.agent_instance = self._agents_client.create_agent(
                model=member.model, name=member.name, instructions=extended_instructions, toolset=member.toolset, tools=member.tools, tool_resources=member.tool_resources
            )
            
            # Emit agent created event
            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                event_emitter.emit_sync("agent_created", "agent", {
                    "agent_name": member.name,
                    "model": member.model,
                    "instructions": member.instructions[:200] + "..." if len(member.instructions) > 200 else member.instructions,
                    "agent_type": "team_member",
                    "can_delegate": member.can_delegate
                })

    def dismantle_team(self) -> None:
        """
        Delete all agents (including the team leader) from the project client.
        """
        assert self._agents_client is not None, "agents_client must not be None"

        if self._team_leader and self._team_leader.agent_instance:
            print(f"Deleting team leader agent '{self._team_leader.name}'")
            self._agents_client.delete_agent(self._team_leader.agent_instance.id)
        for member in self._members:
            if member is not self._team_leader and member.agent_instance:
                print(f"Deleting agent '{member.name}'")
                self._agents_client.delete_agent(member.agent_instance.id)
        AgentTeam._remove_team(self.team_name)

    def _add_task_completion_event(
        self,
        span: Span,
        result: str,
    ) -> None:

        attributes: Dict[str, Any] = {}
        attributes["agent_team.task.result"] = result
        span.add_event(name=f"agent_team.task_completed", attributes=attributes)

    def process_request(self, request: str) -> str:
        """
        Handle a user's request by creating a team and delegating tasks to
        the team leader. The team leader may generate additional tasks.

        :param request: The user's request or question.
        :return: Structured markdown response from the agent team
        """
        assert self._agents_client is not None, "project client must not be None"
        assert self._team_leader is not None, "team leader must not be None"

        agent_responses = []

        if self._agent_thread is None:
            self._agent_thread = self._agents_client.threads.create()
            print(f"Created thread with ID: {self._agent_thread.id}")

        with tracer.start_as_current_span("agent_team_request") as current_request_span:
            self._current_request_span = current_request_span
            if self._current_request_span is not None:
                self._current_request_span.set_attribute("agent_team.name", self.team_name)
            
            # Emit team processing started event
            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                event_emitter.emit_sync("processing_started", "team", {
                    "team_name": self.team_name,
                    "request": request,
                    "thread_id": self._agent_thread.id if self._agent_thread else None
                })
            
            team_leader_request = self.TEAM_LEADER_INITIAL_REQUEST.format(original_request=request)
            _create_task(
                team_name=self.team_name,
                recipient=self._team_leader.name,
                request=team_leader_request,
                requestor="user",
            )
            while self._tasks:
                task = self._tasks.pop(0)
                with tracer.start_as_current_span("agent_team_task") as current_task_span:
                    self._current_task_span = current_task_span
                    if self._current_task_span is not None:
                        self._current_task_span.set_attribute("agent_team.name", self.team_name)
                        self._current_task_span.set_attribute("agent_team.task.recipient", task.recipient)
                        self._current_task_span.set_attribute("agent_team.task.requestor", task.requestor)
                        self._current_task_span.set_attribute("agent_team.task.description", task.task_description)
                    print(
                        f"Starting task for agent '{task.recipient}'. "
                        f"Requestor: '{task.requestor}'. "
                        f"Task description: '{task.task_description}'."
                    )
                    
                    # Emit task started event
                    if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                        event_emitter.emit_sync("task_started", "task", {
                            "recipient": task.recipient,
                            "requestor": task.requestor,
                            "task_description": task.task_description,
                            "task_id": f"{task.recipient}_{len(agent_responses)}"
                        })
                    
                    # Emit agent started task event
                    if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                        event_emitter.emit_sync("agent_started_task", "agent", {
                            "agent_name": task.recipient,
                            "task_description": task.task_description
                        })
                    
                    # Emit message sent event
                    if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                        event_emitter.emit_sync("message_sent", "task", {
                            "from": task.requestor,
                            "to": task.recipient,
                            "message": task.task_description,
                            "message_id": f"msg_{task.recipient}_{len(agent_responses)}"
                        })
                    
                    message = self._agents_client.messages.create(
                        thread_id=self._agent_thread.id,
                        role="user",
                        content=task.task_description,
                    )
                    print(f"Created message with ID: {message.id} for task in thread {self._agent_thread.id}")
                    agent = self._get_member_by_name(task.recipient)
                    if agent and agent.agent_instance:
                        run = self._agents_client.runs.create_and_process(
                            thread_id=self._agent_thread.id, agent_id=agent.agent_instance.id
                        )
                        print(f"Created and processed run for agent '{agent.name}', run ID: {run.id}")
                        text_message = self._agents_client.messages.get_last_message_text_by_role(
                            thread_id=self._agent_thread.id, role=MessageRole.AGENT
                        )
                        if text_message and text_message.text:
                            agent_response_text = text_message.text.value
                            print(f"Agent '{agent.name}' completed task. Outcome: {agent_response_text}")
                            
                            # Emit response generated event
                            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                                event_emitter.emit_sync("response_generated", "task", {
                                    "agent": agent.name,
                                    "response": agent_response_text,
                                    "original_message": task.task_description,
                                    "message_id": f"resp_{task.recipient}_{len(agent_responses)}"
                                })
                            
                            # Emit task completed event
                            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                                event_emitter.emit_sync("task_completed", "task", {
                                    "recipient": task.recipient,
                                    "task_description": task.task_description,
                                    "result": agent_response_text[:500] + "..." if len(agent_response_text) > 500 else agent_response_text,
                                    "task_id": f"{task.recipient}_{len(agent_responses)}"
                                })
                            
                            # Emit agent completed task event
                            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                                event_emitter.emit_sync("agent_completed_task", "agent", {
                                    "agent_name": agent.name,
                                    "task_result": agent_response_text[:200] + "..." if len(agent_response_text) > 200 else agent_response_text
                                })
                            
                            # Collect agent response for markdown formatting
                            agent_responses.append({
                                "agent": agent.name,
                                "response": agent_response_text,
                                "task": task.task_description
                            })
                            
                            if self._current_task_span is not None:
                                self._add_task_completion_event(self._current_task_span, result=agent_response_text)

                    # If no tasks remain AND the recipient is not the TeamLeader,
                    # let the TeamLeader see if more delegation is needed.
                    if not self._tasks and not task.recipient == "TeamLeader":
                        team_leader_request = self.TEAM_LEADER_TASK_COMPLETENESS_CHECK_INSTRUCTIONS
                        _create_task(
                            team_name=self.team_name,
                            recipient=self._team_leader.name,
                            request=team_leader_request,
                            requestor="user",
                        )
                    self._current_task_span = None
            self._current_request_span = None
            
            # Format and return structured markdown response
            thread_id = self._agent_thread.id if self._agent_thread else "Unknown"
            markdown_response = self._format_markdown_response(request, agent_responses, thread_id)
            
            # Emit team processing completed event
            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                event_emitter.emit_sync("processing_completed", "team", {
                    "team_name": self.team_name,
                    "request": request,
                    "final_result": markdown_response,
                    "result": markdown_response,
                    "response_length": len(markdown_response),
                    "agents_involved": len(agent_responses),
                    "thread_id": thread_id,
                    "summary": f"Completed analysis with {len(agent_responses)} agent responses"
                })
            
            # Print the formatted markdown to console
            print("\n" + "="*80)
            print("AGENT TEAM MARKDOWN RESPONSE:")
            print("="*80)
            print(markdown_response)
            print("="*80 + "\n")
            
            return markdown_response

    def process_request_with_results(self, request: str) -> Dict[str, Any]:
        """
        Handle a user's request and return the conversation results.

        :param request: The user's request or question.
        :return: Dictionary containing conversation messages and metadata
        """
        assert self._agents_client is not None, "project client must not be None"
        assert self._team_leader is not None, "team leader must not be None"

        if self._agent_thread is None:
            self._agent_thread = self._agents_client.threads.create()
            print(f"Created thread with ID: {self._agent_thread.id}")

        conversation_messages = []
        
        with tracer.start_as_current_span("agent_team_request") as current_request_span:
            self._current_request_span = current_request_span
            if self._current_request_span is not None:
                self._current_request_span.set_attribute("agent_team.name", self.team_name)
            
            team_leader_request = self.TEAM_LEADER_INITIAL_REQUEST.format(original_request=request)
            _create_task(
                team_name=self.team_name,
                recipient=self._team_leader.name,
                request=team_leader_request,
                requestor="user",
            )
            
            while self._tasks:
                task = self._tasks.pop(0)
                with tracer.start_as_current_span("agent_team_task") as current_task_span:
                    self._current_task_span = current_task_span
                    if self._current_task_span is not None:
                        self._current_task_span.set_attribute("agent_team.name", self.team_name)
                        self._current_task_span.set_attribute("agent_team.task.recipient", task.recipient)
                        self._current_task_span.set_attribute("agent_team.task.requestor", task.requestor)
                        self._current_task_span.set_attribute("agent_team.task.description", task.task_description)
                    
                    print(
                        f"Starting task for agent '{task.recipient}'. "
                        f"Requestor: '{task.requestor}'. "
                        f"Task description: '{task.task_description}'."
                    )
                    
                    # Emit message processing event
                    if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                        event_emitter.emit_sync("message_processing", "task", {
                            "agent": task.recipient,
                            "message": task.task_description,
                            "requestor": task.requestor,
                            "status": "processing"
                        })
                    
                    # Add task info to conversation
                    conversation_messages.append({
                        "role": "user",
                        "agent": task.requestor,
                        "recipient": task.recipient,
                        "content": task.task_description,
                        "message_type": "task_assignment"
                    })
                    
                    # Emit message sent event  
                    if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                        event_emitter.emit_sync("message_sent", "task", {
                            "from": task.requestor,
                            "to": task.recipient,
                            "message": task.task_description,
                            "message_id": f"msg_simple_{task.recipient}_{len(conversation_messages)}"
                        })
                    
                    message = self._agents_client.messages.create(
                        thread_id=self._agent_thread.id,
                        role="user",
                        content=task.task_description,
                    )
                    print(f"Created message with ID: {message.id} for task in thread {self._agent_thread.id}")
                    
                    agent = self._get_member_by_name(task.recipient)
                    if agent and agent.agent_instance:
                        run = self._agents_client.runs.create_and_process(
                            thread_id=self._agent_thread.id, agent_id=agent.agent_instance.id
                        )
                        print(f"Created and processed run for agent '{agent.name}', run ID: {run.id}")
                        text_message = self._agents_client.messages.get_last_message_text_by_role(
                            thread_id=self._agent_thread.id, role=MessageRole.AGENT
                        )
                        if text_message and text_message.text:
                            agent_response = text_message.text.value
                            print(f"Agent '{agent.name}' completed task. Outcome: {agent_response}")
                            
                            # Emit response generated event
                            if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                                event_emitter.emit_sync("response_generated", "task", {
                                    "agent": agent.name,
                                    "response": agent_response,
                                    "original_message": task.task_description,
                                    "message_id": f"resp_simple_{task.recipient}_{len(conversation_messages)}"
                                })
                            
                            # Add agent response to conversation
                            conversation_messages.append({
                                "role": "agent",
                                "agent": agent.name,
                                "content": agent_response,
                                "message_type": "agent_response"
                            })
                            
                            if self._current_task_span is not None:
                                self._add_task_completion_event(self._current_task_span, result=agent_response)

                    # If no tasks remain AND the recipient is not the TeamLeader,
                    # let the TeamLeader see if more delegation is needed.
                    if not self._tasks and not task.recipient == "TeamLeader":
                        team_leader_request = self.TEAM_LEADER_TASK_COMPLETENESS_CHECK_INSTRUCTIONS
                        _create_task(
                            team_name=self.team_name,
                            recipient=self._team_leader.name,
                            request=team_leader_request,
                            requestor="user",
                        )
                    
                    self._current_task_span = None
            
            self._current_request_span = None

        # Get the final conversation state
        final_response = ""
        if conversation_messages:
            # Get the last agent response as the main response
            agent_responses = [msg for msg in conversation_messages if msg["message_type"] == "agent_response"]
            if agent_responses:
                final_response = agent_responses[-1]["content"]
                
                # Emit final result for simple processing
                if WEBSOCKET_EVENTS_AVAILABLE and event_emitter:
                    event_emitter.emit_sync("processing_completed", "team", {
                        "team_name": self.team_name,
                        "final_result": final_response,
                        "result": final_response,
                        "response_length": len(final_response),
                        "agents_involved": len(agent_responses),
                        "thread_id": self._agent_thread.id if self._agent_thread else "Unknown",
                        "summary": f"Simple processing completed with {len(agent_responses)} agent responses"
                    })

        #return {
        #    "response": final_response,
        #    "conversation": conversation_messages,
        #    "team_name": self.team_name
        ##    "thread_id": self._agent_thread.id if self._agent_thread else None,
        #}

    def _get_member_by_name(self, name) -> Optional[_AgentTeamMember]:
        """
        Retrieve a team member (agent) by name.
        If no member with the specified name is found, returns None.

        :param name: The agent's name within this team.
        """
        if name == "TeamLeader":
            return self._team_leader
        for member in self._members:
            if member.name == name:
                return member
        return None

    def _format_markdown_response(self, request: str, agent_responses: list, thread_id: str) -> str:
        """
        Format agent responses as structured markdown.
        
        Args:
            request: The original user request
            agent_responses: List of agent response dictionaries
            thread_id: The conversation thread ID
            
        Returns:
            str: Formatted markdown response
        """
        from datetime import datetime
        
        # Start with header
        markdown = f"""# Agent Team Analysis Report

## 📋 Request Summary
**Query:** {request}
**Team:** {self.team_name}
**Thread ID:** {thread_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🤖 Agent Collaboration
"""
        
        # Add each agent's contribution
        for i, response_data in enumerate(agent_responses, 1):
            agent_name = response_data.get('agent', 'Unknown Agent')
            response_text = response_data.get('response', 'No response')
            
            # Determine agent emoji based on name
            emoji = "🔍"  # Default
            if "RAG" in agent_name or "Search" in agent_name:
                emoji = "📊"
            elif "KG" in agent_name or "Graph" in agent_name:
                emoji = "🕸️"
            elif "Bing" in agent_name or "Web" in agent_name:
                emoji = "🌐"
            elif "Leader" in agent_name:
                emoji = "👨‍💼"
            
            markdown += f"""
### {emoji} {agent_name} (Step {i})

```markdown
{response_text}
```

---
"""
        
        # Add summary section if we have responses
        if agent_responses:
            final_response = agent_responses[-1].get('response', '')
            markdown += f"""
## 📋 Executive Summary

{final_response}

---

## 📊 Process Metrics
- **Total Agents Involved:** {len(agent_responses)}
- **Processing Steps:** {len(agent_responses)}
- **Team Coordination:** Multi-agent collaboration completed successfully
"""
        
        return markdown

    """
    Requests another agent in the team to complete a task.

    :param span (Span): The event will be added to this span
    :param team_name (str): The name of the team.
    :param recipient (str): The name of the agent that is being requested to complete the task.
    :param request (str): A description of the to complete. This can also be a question.
    :param requestor (str): The name of the agent who is requesting the task.
    :return: True if the task was successfully received, False otherwise.
    :rtype: str
    """


def _add_create_task_event(
    span: Span,
    team_name: str,
    requestor: str,
    recipient: str,
    request: str,
) -> None:

    attributes: Dict[str, Any] = {}
    attributes["agent_team.task.team_name"] = team_name
    attributes["agent_team.task.requestor"] = requestor
    attributes["agent_team.task.recipient"] = recipient
    attributes["agent_team.task.description"] = request
    span.add_event(name=f"agent_team.create_task", attributes=attributes)


def _create_task(team_name: str, recipient: str, request: str, requestor: str) -> str:
    """
    Requests another agent in the team to complete a task.

    :param team_name (str): The name of the team.
    :param recipient (str): The name of the agent that is being requested to complete the task.
    :param request (str): A description of the to complete. This can also be a question.
    :param requestor (str): The name of the agent who is requesting the task.
    :return: True if the task was successfully received, False otherwise.
    :rtype: str
    """
    task = AgentTask(recipient=recipient, task_description=request, requestor=requestor)
    team: Optional[AgentTeam] = None
    try:
        team = AgentTeam.get_team(team_name)
        span: Optional[Span] = None
        if team._current_task_span is not None:
            span = team._current_task_span
        elif team._current_request_span is not None:
            span = team._current_request_span

        if span is not None:
            _add_create_task_event(
                span=span, team_name=team_name, requestor=requestor, recipient=recipient, request=request
            )
    except:
        pass
    if team is not None:
        team.add_task(task)
        return "True"
    return "False"


# Any additional functions that might be used by the agents:
agent_team_default_functions: Set = {
    _create_task,
}

default_function_tool = FunctionTool(functions=agent_team_default_functions)

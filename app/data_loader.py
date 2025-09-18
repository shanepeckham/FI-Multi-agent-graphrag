import os
import json
from pathlib import Path
from typing import List, Dict, Any


def load_actions() -> List[Dict[str, Any]]:
    """
    Load available actions from the actions.json file.

    Returns:
        List of action dictionaries, each containing:
        - task_type: The action type (e.g., "schedule_meeting")
        - parameters: Dictionary of parameter names and their types
    """
    actions_path = "./actions.json"
    with open(actions_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_languages() -> List[Dict[str, Any]]:
    """
    Load available actions from the actions.json file.

    Returns:
        List of language dictionaries, each containing:
        - language_name: The language name (e.g., "English")
        - language_name_abbrev: The language name abbreviation (e.g., "EN")
    """
    languages_path = "./languages.json"
    with open(languages_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_action_types() -> List[str]:
    """
    Get a list of available action types.

    Returns:
        List of action type strings
    """
    actions = load_actions()
    return [action["task_type"] for action in actions]

def get_languages() -> List[str]:
    """
    Get a list of available languages.

    Returns:
        List of language strings
    """
    languages = load_languages()
    return [language["language_name"] for language in languages]


def get_action_parameters(task_type: str) -> Dict[str, str] | None:
    """
    Get the parameters for a specific action type.

    Args:
        task_type: The task type to get parameters for

    Returns:
        Dictionary of parameter names and their types, None if action not found
    """
    actions = load_actions()
    for action in actions:
        if action["task_type"] == task_type:
            return action.get("parameters", {})
    return None


def get_all_action_parameters() -> Dict[str, Dict[str, str]]:
    """
    Get all action types with their parameters.

    Returns:
        Dictionary mapping action types to their parameters
    """
    actions = load_actions()
    return {action["task_type"]: action.get("parameters", {}) for action in actions}

"""JSON parsing utilities for training data generation.

Handles malformed JSON responses from teacher models, particularly
issues with unescaped newlines in string values.
"""

import json
import re
from typing import Any, Optional


def extract_json_from_response(response: str) -> Optional[dict[str, Any]]:
    """Extract and parse JSON from LLM response, with robust error handling.

    Args:
        response: Raw LLM response that may contain JSON

    Returns:
        Parsed JSON dict, or None if parsing fails

    Common issues this handles:
    - JSON wrapped in markdown code blocks
    - Unescaped newlines in string values
    - Leading/trailing whitespace
    - Missing outer braces
    """
    if not response or not isinstance(response, str):
        return None

    # Step 1: Extract JSON from markdown code blocks
    json_text = response.strip()

    if "```json" in json_text:
        # Extract from ```json ... ```
        parts = json_text.split("```json", 1)
        if len(parts) > 1:
            json_text = parts[1].split("```", 1)[0].strip()
    elif "```" in json_text:
        # Extract from ``` ... ```
        parts = json_text.split("```", 1)
        if len(parts) > 1:
            json_text = parts[1].split("```", 1)[0].strip()
    elif "{" in json_text:
        # Extract from first { to last }
        start = json_text.find("{")
        end = json_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = json_text[start : end + 1]

    # Step 2: Try standard parsing first
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Step 3: Attempt to fix common issues

    # Fix unescaped newlines in string values
    # This regex finds string values and escapes newlines within them
    def fix_string_newlines(match):
        """Replace literal newlines with \\n in JSON string values."""
        string_content = match.group(1)
        # Replace literal newlines with escaped newlines
        fixed = string_content.replace('\n', '\\n').replace('\r', '\\r')
        return f'"{fixed}"'

    try:
        # Match string values in JSON (simplified - handles most cases)
        # Pattern: "..." where ... can contain escaped quotes \" but not bare "
        fixed_json = re.sub(
            r'"([^"\\]*(?:\\.[^"\\]*)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            json_text,
            flags=re.DOTALL
        )

        return json.loads(fixed_json)
    except (json.JSONDecodeError, re.error):
        pass

    # Step 4: Try a more aggressive fix - manually parse key-value pairs
    try:
        # Extract instruction, input, output fields manually
        result = {}

        # Find "instruction": "..."
        inst_match = re.search(r'"instruction"\s*:\s*"(.*?)"(?=\s*,|\s*})', json_text, re.DOTALL)
        if inst_match:
            result["instruction"] = inst_match.group(1).replace('\n', ' ').strip()

        # Find "input": "..."
        input_match = re.search(r'"input"\s*:\s*"(.*?)"(?=\s*,|\s*})', json_text, re.DOTALL)
        if input_match:
            result["input"] = input_match.group(1).replace('\n', ' ').strip()

        # Find "output": "..." (may span multiple lines)
        output_match = re.search(r'"output"\s*:\s*"(.*?)"\s*}', json_text, re.DOTALL)
        if output_match:
            result["output"] = output_match.group(1).strip()

        if result:
            return result
    except Exception:
        pass

    return None


def validate_training_sample_json(data: dict[str, Any]) -> bool:
    """Validate that parsed JSON has required fields for training sample.

    Args:
        data: Parsed JSON dict

    Returns:
        True if valid training sample JSON
    """
    if not isinstance(data, dict):
        return False

    # Must have instruction, input, output
    required_fields = ["instruction", "input", "output"]

    for field in required_fields:
        if field not in data:
            return False

        # Fields should be non-empty strings
        value = data[field]
        if not isinstance(value, str):
            return False

        # Instruction and output must be non-empty
        if field in ("instruction", "output") and not value.strip():
            return False

    return True

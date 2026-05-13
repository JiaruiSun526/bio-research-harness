"""Tool registry — registers Python functions as LLM-callable tools.

Tools are plain Python functions. The registry:
1. Stores their metadata (name, description, JSON Schema for parameters)
2. Generates OpenAI function-calling format definitions for LLM calls
3. Dispatches execution by name, catching ToolError → error tool results
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import TypeAdapter, ValidationError


class ToolError(Exception):
    """Raised by tool implementations when preconditions fail or arguments are invalid.

    ToolRegistry.execute() catches this and returns (error_message, is_error=True)
    so the LLM sees the error and can adjust its strategy.
    """


@dataclass(frozen=True)
class ToolDefinition:
    """Internal representation of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    terminal: bool = False


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured tool execution result used by the agentic loop."""

    content: str
    is_error: bool
    is_terminal: bool = False


class ToolRegistry:
    """Registry of tools available to an agent.

    Usage pattern (via factory function in tools.py):

        registry = ToolRegistry()

        @registry.register(
            name="save_plan",
            description="Save a research plan draft for a stage.",
            parameters={
                "stage_id": {"type": "string", "description": "Stage identifier"},
                "content": {"type": "string", "description": "Plan content in Markdown"},
            },
        )
        def save_plan(stage_id: str, content: str) -> str:
            workspace.write_plan(stage_id, content)
            return f"Plan saved for {stage_id}."
    """

    def __init__(self) -> None:
        """Initialize an empty registry keyed by tool name."""

        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Decorator that registers a function as a tool.

        Args:
            name: Tool name (identifier the LLM uses to call it).
            description: Tool description (the LLM reads this to decide when to call).
            parameters: Dict mapping parameter names to JSON Schema property objects.
                Each value should have at minimum {"type": ..., "description": ...}.

        Returns:
            Decorator that registers the function and returns it unchanged.
        """

        def decorator(handler: Callable[..., str]) -> Callable[..., str]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
                terminal=terminal,
            )
            return handler

        return decorator

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool call by name and return the legacy tuple interface."""

        result = self.execute_with_metadata(name, arguments)
        return (result.content, result.is_error)

    def execute_with_metadata(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool call by name.

        Args:
            name: Tool name.
            arguments: Arguments dict parsed from the LLM's tool call.

        Returns:
            (result_content, is_error):
            - On success: (result string, False)
            - On ToolError: (error message, True)
            - On unknown tool: (error message, True)
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(content=f"Unknown tool: {name}", is_error=True)

        try:
            signature = inspect.signature(tool.handler)
            try:
                bound_arguments = signature.bind(**arguments)
            except TypeError as error:
                raise ToolError(f"Invalid arguments for '{name}': {error}") from error
            validated_arguments = _validate_bound_arguments(tool.handler, bound_arguments.arguments)
            result = tool.handler(**validated_arguments)
        except ToolError as error:
            return ToolExecutionResult(content=str(error), is_error=True)

        return ToolExecutionResult(
            content=result,
            is_error=False,
            is_terminal=tool.terminal,
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI function-calling format.

        Each entry is:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { "type": "object", "properties": {...}, "required": [...] }
                }
            }
        """
        definitions: list[dict[str, Any]] = []
        for tool in self._tools.values():
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.parameters,
                            "required": _get_required_parameter_names(tool.handler),
                        },
                    },
                }
            )
        return definitions


def _get_required_parameter_names(handler: Callable[..., str]) -> list[str]:
    """Return handler parameter names that must be supplied by the caller."""

    required_names: list[str] = []
    for parameter in inspect.signature(handler).parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if parameter.default is inspect.Signature.empty:
            required_names.append(parameter.name)
    return required_names


def _validate_bound_arguments(
    handler: Callable[..., str],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Validate bound arguments against the handler's type annotations."""

    validated_arguments: dict[str, Any] = {}
    signature = inspect.signature(handler)
    resolved_hints = get_type_hints(handler)
    for parameter_name, value in arguments.items():
        annotation = resolved_hints.get(
            parameter_name,
            signature.parameters[parameter_name].annotation,
        )
        if annotation is inspect.Signature.empty:
            validated_arguments[parameter_name] = value
            continue
        try:
            validated_arguments[parameter_name] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as error:
            raise ToolError(
                f"Invalid value for '{parameter_name}' in '{handler.__name__}': {error}"
            ) from error
    return validated_arguments

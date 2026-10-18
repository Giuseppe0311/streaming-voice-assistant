import asyncio
import json
from pydantic import ValidationError
from app.tools.registry import ToolRegistry


def reject_nonfinite(value):
    raise ValueError("Non-finite JSON number")


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout: float = 5):
        self.registry = registry
        self.timeout = timeout

    async def execute(self, name: str, arguments: str):
        try:
            tool = self.registry.get(name)
        except KeyError:
            return {'error': 'unknown_tool'}
        try:
            if len(arguments) > 16000:
                return {'error': 'invalid_arguments'}
            data = json.loads(arguments, parse_constant=reject_nonfinite)
            args = tool.arguments.model_validate(data, strict=True)
        except (ValidationError, ValueError, TypeError):
            return {'error': 'invalid_arguments'}
        try:
            async with asyncio.timeout(self.timeout):
                result = await tool.handler(args)
                # Fail closed for custom handlers that return non-JSON data.
                json.dumps(result, allow_nan=False)
                return result
        except TimeoutError:
            return {'error': 'tool_timeout'}
        except Exception:
            return {'error': 'tool_failed'}

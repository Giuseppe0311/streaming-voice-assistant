from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[Any]]

    def schema(self):
        return {'type': 'function', 'name': self.name,
                'description': self.description,
                'parameters': self.arguments.model_json_schema(), 'strict': True}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f'Duplicate tool: {tool.name}')
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools[name]

    def schemas(self):
        return [tool.schema() for tool in self._tools.values()]

import math
import operator
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.tools.registry import Tool, ToolRegistry


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    left: float
    right: float

    @field_validator('left', 'right')
    @classmethod
    def bounded_operand(cls, value: float):
        # Keep extreme bounds out of the provider-facing JSON Schema. Some
        # models exhaust their output budget when those constraints are sent.
        if abs(value) > 1e100:
            raise ValueError('Operand is outside the supported range')
        return value


class DateTimeArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    timezone: str = Field(description='IANA timezone, e.g. America/Lima or UTC', max_length=100)


async def calculate(args: CalculatorArguments):
    operations = {'add': operator.add, 'subtract': operator.sub,
                  'multiply': operator.mul, 'divide': operator.truediv}
    result = operations[args.operation](args.left, args.right)
    if not math.isfinite(result):
        raise ValueError('Non-finite result')
    return {'result': result}


async def current_datetime(args: DateTimeArguments):
    return {'datetime': datetime.now(ZoneInfo(args.timezone)).isoformat(),
            'timezone': args.timezone}


def default_registry():
    registry = ToolRegistry()
    registry.register(Tool('calculator', 'Calculate one arithmetic operation.', CalculatorArguments, calculate))
    registry.register(Tool('current_datetime', 'Get current date and time in an IANA timezone.', DateTimeArguments, current_datetime))
    return registry

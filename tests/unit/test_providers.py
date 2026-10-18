import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from app.config import Settings
from app.providers.llm import LanguageModel
from app.providers.transcription import Transcription
from app.tools.builtins import default_registry
from app.tools.executor import ToolExecutor


def response(text='', calls=()):
    return NS(status='completed', output_text=text, output=list(calls))


class Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    async def create(self, **kwargs):
        self.requests.append(kwargs)
        result = next(self.responses)
        class Stream:
            def __aiter__(self):
                return self.events()
            async def events(self):
                yield NS(type="response.completed", response=result)
            async def close(self):
                pass
        return Stream()


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_speculation_proposes_but_does_not_execute_tools(self):
        registry = default_registry()
        executor = ToolExecutor(registry)
        executed = []
        original = executor.execute
        async def execute(*args):
            executed.append(args)
            return await original(*args)
        executor.execute = execute
        call = NS(type='function_call', name='calculator', arguments='{"operation":"add","left":2,"right":3}', call_id='c1', model_dump=lambda **kwargs: {'type':'function_call','name':'calculator','arguments':'{"operation":"add","left":2,"right":3}','call_id':'c1'})
        api = Responses([response(calls=[call]), response('Five.')])
        llm = LanguageModel(NS(responses=api), Settings(), executor)
        draft = await llm.draft('2 + 3')
        self.assertEqual(executed, [])
        answer = ''.join([text async for text in llm.respond('2 + 3', draft)])
        self.assertEqual(answer, 'Five.')
        self.assertEqual(len(executed), 1)
        self.assertTrue(api.requests[0]['tools'])
        self.assertFalse(api.requests[0]['store'])
        self.assertEqual(api.requests[1]['input'][-1]['type'], 'function_call_output')

    async def test_tool_budget_stops_infinite_model_loop(self):
        call = NS(type='function_call', name='current_datetime', arguments='{"timezone":"UTC"}', call_id='c1', model_dump=lambda **kwargs: {'type':'function_call','name':'current_datetime','arguments':'{"timezone":"UTC"}','call_id':'c1'})
        api = Responses([response(calls=[call]) for _ in range(5)])
        llm = LanguageModel(NS(responses=api), Settings(max_tool_calls=2), ToolExecutor(default_registry()))
        with self.assertRaisesRegex(RuntimeError, 'Tool call limit'):
            _ = [text async for text in llm.respond('time', None)]
        self.assertEqual(len(api.requests), 3)

    async def test_transcription_error_is_not_silently_ignored(self):
        transcriber = Transcription('not-a-real-key', Settings())
        class Socket:
            async def recv(self):
                return json.dumps({'type':'error','error':{'code':'bad_request'}})
        transcriber._ws = Socket()
        with self.assertRaisesRegex(RuntimeError, 'Transcription'):
            await transcriber._receive(lambda text: None)

import asyncio
import struct
import unittest

from app.audio.vad import SilenceDetector
from app.conversation.transcripts import Drafts
from app.tools.builtins import default_registry
from app.tools.executor import ToolExecutor


class SilenceTests(unittest.TestCase):
    def test_initial_silence_and_short_pauses_do_not_end_turn(self):
        vad = SilenceDetector(threshold=0.02, silence_seconds=0.3, min_speech_seconds=0.1, max_turn_seconds=10)
        quiet = bytes(4800)
        loud = struct.pack('<2400h', *([10000] * 2400))
        for _ in range(20):
            self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(loud))
        self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(loud))
        self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(quiet))
        self.assertTrue(vad.feed(quiet))

    def test_soft_fillers_keep_confirmed_turn_open_without_starting_one(self):
        vad = SilenceDetector(threshold=0.02, release_ratio=0.5,
                              silence_seconds=0.3, min_speech_seconds=0.2)
        quiet = bytes(4800)
        filler = struct.pack('<2400h', *([400] * 2400))
        speech = struct.pack('<2400h', *([10000] * 2400))

        for _ in range(5):
            self.assertFalse(vad.feed(filler))
        self.assertFalse(vad.started)

        self.assertFalse(vad.feed(speech))
        self.assertFalse(vad.feed(speech))
        for _ in range(10):
            self.assertFalse(vad.feed(filler))
        self.assertFalse(vad.feed(speech))

        self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(quiet))
        self.assertTrue(vad.feed(quiet))


class DraftTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_exact_latest_draft_can_be_reused(self):
        calls = []
        async def generate(text):
            calls.append(text)
            return text.upper()
        drafts = Drafts(generate, delay=0.01)
        drafts.update('old')
        drafts.update('new')
        await asyncio.sleep(0.03)
        self.assertEqual(calls, ['new'])
        self.assertIsNone(await drafts.finish('changed'))
        drafts.update('exact')
        await asyncio.sleep(0.03)
        self.assertEqual(await drafts.finish('exact'), 'EXACT')
        await drafts.close()

    async def test_cancelled_draft_never_reused(self):
        started = asyncio.Event()
        async def generate(text):
            started.set()
            await asyncio.sleep(10)
        drafts = Drafts(generate, delay=0)
        drafts.update('first')
        await started.wait()
        self.assertIsNone(await drafts.finish('first'))
        await drafts.close()


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_calculator_and_validation(self):
        registry = default_registry()
        executor = ToolExecutor(registry, timeout=0.1)
        result = await executor.execute('calculator', '{"operation":"multiply","left":6,"right":7}')
        self.assertEqual(result, {'result':42})
        for payload in ['{"operation":"divide","left":1,"right":0}', '{"operation":"add","left":true,"right":2}', '{"operation":"add","left":1,"right":2,"code":"bad"}', 'null', '{"operation":"add","left":NaN,"right":2}', '{"operation":"add","left":1e101,"right":2}']:
            self.assertIn('error', await executor.execute('calculator', payload))
        calculator_schema = registry.schemas()[0]['parameters']['properties']
        self.assertNotIn('minimum', calculator_schema['left'])
        self.assertNotIn('maximum', calculator_schema['left'])
        self.assertIn('error', await executor.execute('missing', '{}'))

    async def test_tool_timeout(self):
        from app.tools.registry import Tool, ToolRegistry
        from pydantic import BaseModel
        class Empty(BaseModel):
            pass
        async def slow(args):
            await asyncio.sleep(10)
        registry = ToolRegistry()
        registry.register(Tool('slow', 'Slow test', Empty, slow))
        self.assertEqual((await ToolExecutor(registry, timeout=0.01).execute('slow','{}'))['error'], 'tool_timeout')

class MoreDraftTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_partials_still_start_speculation(self):
        calls = []
        async def generate(text):
            calls.append(text)
            return text
        drafts = Drafts(generate, delay=0.02)
        for index in range(8):
            drafts.update(str(index))
            await asyncio.sleep(0.01)
        await drafts.close()
        self.assertGreaterEqual(len(calls), 2)
        self.assertLess(len(calls), 8)

    async def test_failed_speculation_falls_back_without_unhandled_task(self):
        async def generate(text):
            raise RuntimeError('provider down')
        drafts = Drafts(generate, delay=0)
        drafts.update('hello')
        await asyncio.sleep(0.01)
        self.assertIsNone(await drafts.finish('hello'))


class SegmentTests(unittest.TestCase):
    def test_long_unbroken_text_is_bounded(self):
        from app.providers.llm import speech_segments
        segments = list(speech_segments('x' * 701))
        self.assertEqual([len(segment) for segment in segments], [300, 300, 101])

class FalseStartTests(unittest.TestCase):
    def test_short_noise_resets_before_later_valid_speech(self):
        vad = SilenceDetector(silence_seconds=0.3, min_speech_seconds=0.2)
        loud = struct.pack('<2400h', *([10000] * 2400))
        quiet = bytes(4800)
        self.assertFalse(vad.feed(loud))
        for _ in range(3):
            self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.started)
        for _ in range(100):
            self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(loud))
        self.assertFalse(vad.feed(loud))
        self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.feed(quiet))
        self.assertTrue(vad.feed(quiet))

    def test_unconfirmed_onset_obeys_max_turn_bound(self):
        vad = SilenceDetector(silence_seconds=5, min_speech_seconds=2, max_turn_seconds=3)
        loud = struct.pack('<2400h', *([10000] * 2400))
        quiet = bytes(4800)
        self.assertFalse(vad.feed(loud))
        for _ in range(29):
            self.assertFalse(vad.feed(quiet))
        self.assertFalse(vad.started)

"""Transport ownership must end synchronously with the failed/cancelled turn."""
import asyncio
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from app.audio.playback import Playback
from app.conversation.orchestrator import Conversation
from app.conversation.state import State
from app.providers.speech import Speech


def fixture(write):
    events = []
    class Mic:
        active = False
        async def start(self):
            self.active = True
            events.append('mic.start')
        async def stop(self):
            self.active = False
    class Transcriber:
        async def open(self):
            pass
        async def close(self):
            pass
        async def capture(self, microphone, partial, finalizing):
            return 'hello'
    class Model:
        async def draft(self, text):
            return None
        async def respond(self, text, draft):
            try:
                yield 'Hello.'
            finally:
                events.append('llm.closed')
        def remember(self, user, assistant):
            events.append('remember')
    class Response:
        async def __aenter__(self):
            events.append('http.open')
            return self
        async def __aexit__(self, *args):
            events.append('http.closed')
        async def iter_bytes(self, chunk_size):
            try:
                yield b'\0\0'
                yield b'\0\0'
            finally:
                events.append('bytes.closed')
    class Device:
        def __init__(self, **kwargs):
            pass
        def start(self):
            events.append('device.start')
        def write(self, pcm):
            write(events)
        def stop(self):
            events.append('device.drained')
        def abort(self):
            events.append('device.abort')
        def close(self):
            events.append('device.closed')
    client = NS(audio=NS(speech=NS(with_streaming_response=NS(create=lambda **kwargs: Response()))))
    mic = Mic()
    conversation = Conversation(mic, Transcriber(), Model(), Speech(client, 'fake', 'fake'), Playback())
    return conversation, mic, events, Device


class StreamCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_write_failure_closes_nested_http_and_generators_before_return(self):
        def fail(events):
            events.append('write.failed')
            raise RuntimeError('device failed')
        conversation, mic, events, device = fixture(fail)
        with patch.dict('sys.modules', {'sounddevice': NS(RawOutputStream=device)}):
            await conversation.run()
        self.assertEqual(conversation.state, State.ERROR)
        self.assertFalse(mic.active)
        self.assertEqual(events.count('mic.start'), 1)
        self.assertIn('device.closed', events)
        self.assertIn('http.closed', events)
        self.assertIn('bytes.closed', events)
        self.assertIn('llm.closed', events)
        self.assertNotIn('remember', events)

    async def test_cancel_during_write_waits_for_worker_then_closes_all_owners(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def write(events):
            events.append('write.started')
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=2):
                raise RuntimeError('test release timeout')
            events.append('write.finished')
        conversation, mic, events, device = fixture(write)
        with patch.dict('sys.modules', {'sounddevice': NS(RawOutputStream=device)}):
            task = asyncio.create_task(conversation.run())
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                self.assertNotIn('device.closed', events)
                self.assertFalse(mic.active)
                self.assertEqual(events.count('mic.start'), 1)
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(conversation.state, State.STOPPED)
        self.assertLess(events.index('write.finished'), events.index('device.closed'))
        self.assertIn('http.closed', events)
        self.assertIn('bytes.closed', events)
        self.assertIn('llm.closed', events)
        self.assertNotIn('remember', events)

    async def test_repeated_cancellation_cannot_abandon_an_inflight_write(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def write(events):
            events.append('write.started')
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            events.append('write.finished')
        conversation, mic, events, device = fixture(write)
        with patch.dict('sys.modules', {'sounddevice': NS(RawOutputStream=device)}):
            task = asyncio.create_task(conversation.run())
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                await asyncio.sleep(0.01)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                self.assertNotIn('device.closed', events)
                self.assertFalse(mic.active)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertLess(events.index('write.finished'), events.index('device.closed'))

    async def test_write_failure_after_cancellation_does_not_swallow_cancel(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def write(events):
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            raise RuntimeError('device failed during cancellation')
        conversation, mic, events, device = fixture(write)
        with patch.dict('sys.modules', {'sounddevice': NS(RawOutputStream=device)}):
            task = asyncio.create_task(conversation.run())
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0.01)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn('http.closed', events)
        self.assertFalse(mic.active)

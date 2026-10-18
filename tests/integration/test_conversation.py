import asyncio
import unittest
from app.conversation.orchestrator import Conversation
from app.conversation.state import State


class Microphone:
    def __init__(self, events):
        self.events = events
        self.active = False
    async def start(self):
        self.active = True
        self.events.append('mic.start')
    async def stop(self):
        if self.active:
            self.events.append('mic.stop')
        self.active = False


class Transcriber:
    async def open(self):
        pass
    async def close(self):
        pass
    async def capture(self, microphone, on_partial, on_finalizing=lambda: None):
        on_partial('hello')
        await asyncio.sleep(0.02)
        return 'hello'


class LLM:
    def __init__(self):
        self.history = []
    async def draft(self, text):
        return 'draft'
    async def respond(self, text, draft):
        yield 'Hello.'
    def remember(self, user, assistant):
        self.history.append((user, assistant))


class Speech:
    async def stream(self, text):
        yield b'\0\0'


class Playback:
    def __init__(self, mic, events):
        self.mic, self.events = mic, events
        self.drained = asyncio.Event()
        self.release = asyncio.Event()
    async def play(self, chunks):
        async for chunk in chunks:
            assert not self.mic.active
        self.events.append('playback.draining')
        self.drained.set()
        await self.release.wait()
        self.events.append('playback.done')
    async def close(self):
        self.events.append('playback.close')


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mic_restarts_only_after_audio_has_drained(self):
        events = []
        mic = Microphone(events)
        player = Playback(mic, events)
        llm = LLM()
        conversation = Conversation(mic, Transcriber(), llm, Speech(), player, draft_delay=0)
        task = asyncio.create_task(conversation.turn())
        await player.drained.wait()
        self.assertFalse(mic.active)
        self.assertEqual(conversation.state, State.SPEAKING)
        player.release.set()
        await task
        self.assertEqual(events, ['mic.start','mic.stop','playback.draining','playback.done'])
        self.assertEqual(llm.history, [('hello', 'Hello.')])
        # Next iteration starts capture only after the preceding turn returns.
        self.assertEqual(conversation.state, State.LISTENING)

    async def test_failure_always_stops_microphone(self):
        class Broken(Transcriber):
            async def capture(self, mic, partial, on_finalizing=lambda: None):
                raise RuntimeError('connection lost')
        events = []
        mic = Microphone(events)
        player = Playback(mic, events)
        conversation = Conversation(mic, Broken(), LLM(), Speech(), player)
        await conversation.run()
        self.assertFalse(mic.active)
        self.assertEqual(conversation.state, State.ERROR)
        self.assertIsNotNone(conversation.last_error)
        self.assertEqual(conversation.last_error_stage, 'transcription_capture')

    async def test_shutdown_during_playback_does_not_restart_capture(self):
        events = []
        mic = Microphone(events)
        player = Playback(mic, events)
        conversation = Conversation(mic, Transcriber(), LLM(), Speech(), player)
        task = asyncio.create_task(conversation.run())
        await player.drained.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(mic.active)
        self.assertIn('playback.close', events)
        self.assertEqual(conversation.state, State.STOPPED)

class RepeatedTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_capture_starts_after_first_playback_drain(self):
        events = []
        mic = Microphone(events)
        player = Playback(mic, events)
        player.release.set()
        conversation = Conversation(mic, Transcriber(), LLM(), Speech(), player)
        await conversation.turn()
        await conversation.turn()
        self.assertEqual(events.count('mic.start'), 2)
        first_done = events.index('playback.done')
        second_start = events.index('mic.start', 1)
        self.assertGreater(second_start, first_done)
        self.assertEqual(conversation.turn_number, 2)

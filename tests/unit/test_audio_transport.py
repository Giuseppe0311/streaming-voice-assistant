import asyncio
import json
import struct
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from app.audio.playback import Playback
from app.config import Settings
from app.providers.transcription import Transcription


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_committed_only_after_silence_and_microphone_stop(self):
        events = []
        loud = struct.pack('<2400h', *([10000] * 2400))
        quiet = bytes(4800)
        incoming = asyncio.Queue()
        class Socket:
            async def recv(self):
                return json.dumps(await incoming.get())
            async def send(self, raw):
                event = json.loads(raw)
                events.append(event['type'])
                if event['type'] == 'input_audio_buffer.commit':
                    incoming.put_nowait({'type':'input_audio_buffer.committed','item_id':'one'})
                    incoming.put_nowait({'type':'conversation.item.input_audio_transcription.completed', 'item_id':'one', 'transcript':'final text'})
        class Mic:
            active = True
            frames = iter([quiet, loud, loud, quiet, quiet])
            async def read(self):
                return next(self.frames)
            async def stop(self):
                if self.active:
                    events.append('mic.stop')
                    self.active = False
        stt = Transcription('fake', Settings(silence_seconds=0.2))
        stt._ws = Socket()
        text = await stt.capture(Mic(), lambda text: None)
        self.assertEqual(text, 'final text')
        self.assertLess(events.index('mic.stop'), events.index('input_audio_buffer.commit'))
        self.assertEqual(events.count('input_audio_buffer.commit'), 1)

    async def test_missing_final_times_out_and_stops_capture(self):
        class Socket:
            async def recv(self):
                await asyncio.Future()
            async def send(self, raw):
                pass
        class Mic:
            active = True
            frames = iter([struct.pack('<2400h', *([10000] * 2400))] * 2 + [bytes(4800)] * 2)
            async def read(self):
                return next(self.frames)
            async def stop(self):
                self.active = False
        stt = Transcription('fake', Settings(silence_seconds=0.2, request_timeout=0.01))
        stt._ws = Socket()
        mic = Mic()
        with self.assertRaises(TimeoutError):
            await stt.capture(mic, lambda text: None)
        self.assertFalse(mic.active)

    async def test_provider_failure_while_idle_stops_mic(self):
        class Socket:
            async def recv(self):
                return json.dumps({'type':'error'})
        class Mic:
            active = True
            async def read(self):
                await asyncio.Future()
            async def stop(self):
                self.active = False
        stt = Transcription('fake', Settings())
        stt._ws = Socket()
        mic = Mic()
        with self.assertRaises(RuntimeError):
            await stt.capture(mic, lambda text: None)
        self.assertFalse(mic.active)


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_drains_after_last_aligned_sample_before_close(self):
        events = []
        class Stream:
            def __init__(self, **kwargs):
                self.options = kwargs
            def start(self):
                events.append('start')
            def write(self, pcm):
                self.assert_pcm = pcm
                assert len(pcm) % 2 == 0
                events.append(('write', pcm))
            def stop(self):
                events.append('drain')
            def abort(self):
                events.append('abort')
            def close(self):
                events.append('close')
        async def chunks():
            yield b'\x01'
            yield b'\x02\x03\x04'
        with patch.dict('sys.modules', {'sounddevice':NS(RawOutputStream=Stream)}):
            await Playback().play(chunks())
        self.assertEqual(events, ['start', ('write', b'\x01\x02\x03\x04'), 'drain', 'abort', 'close'])

class FinalizationStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalizing_notification_precedes_commit(self):
        order = []
        incoming = asyncio.Queue()
        class Socket:
            async def recv(self):
                return json.dumps(await incoming.get())
            async def send(self, raw):
                if json.loads(raw)['type'] == 'input_audio_buffer.commit':
                    order.append('commit')
                    incoming.put_nowait({'type':'conversation.item.input_audio_transcription.completed','item_id':'one','transcript':'ok'})
        class Mic:
            frames = iter([struct.pack('<2400h', *([10000] * 2400))] * 2 + [bytes(4800)] * 2)
            async def read(self):
                return next(self.frames)
            async def stop(self):
                pass
        stt = Transcription('fake', Settings(silence_seconds=0.2))
        stt._ws = Socket()
        await stt.capture(Mic(), lambda text: None, lambda: order.append('finalizing'))
        self.assertEqual(order, ['finalizing', 'commit'])

class FalseStartTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_noise_is_never_sent_and_later_valid_onset_is_preserved(self):
        import base64
        noise = struct.pack('<2400h', *([9000] * 2400))
        speech = struct.pack('<2400h', *([10000] * 2400))
        quiet = bytes(4800)
        sent = []
        incoming = asyncio.Queue()
        class Socket:
            async def recv(self):
                return json.dumps(await incoming.get())
            async def send(self, raw):
                event = json.loads(raw)
                if event['type'] == 'input_audio_buffer.append':
                    sent.append(base64.b64decode(event['audio']))
                elif event['type'] == 'input_audio_buffer.commit':
                    incoming.put_nowait({'type':'conversation.item.input_audio_transcription.completed',
                                         'item_id':'valid', 'transcript':'valid speech'})
        class Mic:
            frames = iter([noise] + [quiet] * 100 + [speech] * 2 + [quiet] * 3)
            async def read(self):
                return next(self.frames)
            async def stop(self):
                pass
        stt = Transcription('fake', Settings(silence_seconds=0.3))
        stt._ws = Socket()
        self.assertEqual(await stt.capture(Mic(), lambda text: None), 'valid speech')
        self.assertEqual(sent.count(noise), 0)
        self.assertEqual(sent.count(speech), 2)
        self.assertLessEqual(len(sent), 8)

"""Realtime transcription transport with client-side turn boundaries."""
import asyncio
import base64
from collections import deque
import json
import math
from app.audio.vad import SilenceDetector
from app.config import Settings


class Transcription:
    URL = 'wss://api.openai.com/v1/realtime?intent=transcription'

    def __init__(self, api_key: str, settings: Settings):
        self.api_key, self.settings = api_key, settings
        self._ws = None
        self._item_id = None

    async def open(self):
        import websockets
        self._item_id = None
        self._ws = await websockets.connect(
            self.URL, additional_headers={'Authorization': f'Bearer {self.api_key}'},
            open_timeout=self.settings.request_timeout, close_timeout=3,
            max_size=2**20, max_queue=16,
        )
        await self._send({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'transcription': {'model': self.settings.transcription_model,
                                  'languages': ['es', 'en'], 'delay': 'low'},
                'turn_detection': None,
            }},
        }})
        async with asyncio.timeout(self.settings.request_timeout):
            while True:
                event = json.loads(await self._ws.recv())
                if event['type'] == 'session.updated':
                    break
                self._check_error(event)

    async def _send(self, event):
        async with asyncio.timeout(self.settings.request_timeout):
            await self._ws.send(json.dumps(event))

    @staticmethod
    def _check_error(event):
        if event.get('type') in {'error', 'conversation.item.input_audio_transcription.failed'}:
            # Do not include server payloads: they may contain transcript content.
            raise RuntimeError('Transcription provider rejected the request')

    async def _receive(self, on_partial):
        text = ''
        while True:
            event = json.loads(await self._ws.recv())
            self._check_error(event)
            kind = event.get('type', '')
            item_id = event.get('item_id')
            if kind == 'input_audio_buffer.committed':
                if self._item_id is not None and item_id != self._item_id:
                    raise RuntimeError('Transcription item mismatch')
                self._item_id = item_id
            elif kind.startswith('conversation.item.input_audio_transcription.'):
                if self._item_id is None:
                    self._item_id = item_id
                if item_id != self._item_id:
                    continue
                if kind.endswith('.delta'):
                    text += event.get('delta', '')
                    on_partial(text)
                elif kind.endswith('.completed'):
                    return event.get('transcript', '').strip()

    async def capture(self, microphone, on_partial, on_finalizing=lambda: None):
        settings = self.settings
        vad = SilenceDetector(settings.vad_threshold, settings.silence_seconds,
                              settings.min_speech_seconds, settings.max_turn_seconds,
                              settings.vad_release_ratio)
        receiver = asyncio.create_task(self._receive(on_partial))
        async def send_audio():
            # Keep candidate speech local until confirmed. A discarded false
            # start therefore cannot contaminate the server's audio buffer.
            # Input frames are 100 ms; capacity covers the configured turn cap.
            onset = deque(maxlen=math.ceil(settings.max_turn_seconds * 10) + 3)
            streaming = False
            while True:
                pcm = await microphone.read()
                ended = vad.feed(pcm)
                if not streaming:
                    if vad.discarded:
                        onset.clear()
                    onset.append(pcm)
                    if not vad.started:
                        while len(onset) > 3:
                            onset.popleft()
                    if not vad.confirmed:
                        continue
                    while onset:
                        await self._send_audio(onset.popleft())
                    streaming = True
                else:
                    await self._send_audio(pcm)
                if ended:
                    on_finalizing()
                    await microphone.stop()
                    return
        sender = asyncio.create_task(send_audio())
        try:
            done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                # A server failure/early close must stop capture, even in silence.
                receiver.result()
                raise RuntimeError('Transcription completed before client commit')
            await sender
            await self._send({'type': 'input_audio_buffer.commit'})
            return await asyncio.wait_for(receiver, timeout=settings.request_timeout)
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
            await microphone.stop()

    async def _send_audio(self, pcm):
        await self._send({'type': 'input_audio_buffer.append',
                          'audio': base64.b64encode(pcm).decode('ascii')})

    async def close(self):
        if self._ws is not None:
            ws, self._ws = self._ws, None
            await ws.close()

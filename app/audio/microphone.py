import asyncio
import queue
import threading
from app.audio.io import blocking


class Microphone:
    def __init__(self, device=None):
        self.device = device
        self._stream = None
        self._queue = queue.Queue(maxsize=30)
        self._overflow = threading.Event()
        self._active = False

    async def start(self):
        import sounddevice as sd
        self._clear()
        self._overflow.clear()
        self._active = True

        def callback(data, frames, time, status):
            if not self._active:
                return
            if status:
                self._overflow.set()
            try:
                self._queue.put_nowait(bytes(data))
            except queue.Full:
                self._overflow.set()

        self._stream = sd.RawInputStream(
            samplerate=24000, channels=1, dtype='int16', blocksize=2400,
            device=self.device, callback=callback,
        )
        try:
            await blocking(self._stream.start)
        except BaseException:
            await self.stop()
            raise

    async def read(self):
        while self._active:
            if self._overflow.is_set():
                raise RuntimeError('Microphone buffer overflow; restart voice capture')
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
        raise RuntimeError('Microphone is stopped')

    async def stop(self):
        self._active = False
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                await blocking(stream.abort)
            finally:
                await blocking(stream.close)
        self._clear()

    def _clear(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

from contextlib import aclosing
from app.audio.io import blocking


class Playback:
    def __init__(self, device=None):
        self.device = device
        self._stream = None

    async def play(self, chunks):
        # A consumer owns early-close, including device creation/write failures.
        async with aclosing(chunks):
            await self._play(chunks)

    async def _play(self, chunks):
        import sounddevice as sd
        self._stream = sd.RawOutputStream(
            samplerate=24000, channels=1, dtype='int16', device=self.device,
        )
        try:
            await blocking(self._stream.start)
            pending = b''
            async for chunk in chunks:
                pending += chunk
                complete = len(pending) - len(pending) % 2
                if complete:
                    await blocking(self._stream.write, pending[:complete])
                    pending = pending[complete:]
            if pending:
                raise RuntimeError('Truncated PCM sample')
            # PortAudio stop waits for buffered output to finish; abort does not.
            await blocking(self._stream.stop)
        finally:
            await self.close()

    async def close(self):
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                await blocking(stream.abort)
            finally:
                await blocking(stream.close)

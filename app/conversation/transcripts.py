"""One replaceable speculative draft; no history or side effects here."""
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class Drafts:
    def __init__(self, generate: Callable[[str], Awaitable[Any]], delay: float):
        self.generate = generate
        self.delay = delay
        self.text = ''
        self.version = 0
        self._task = None
        self._wake = asyncio.Event()
        self._worker = None
        self._result = None

    def update(self, text: str) -> None:
        if text == self.text or not text.strip():
            return
        self.text = text
        self.version += 1
        self._result = None
        if self._task:
            self._task.cancel()
        self._wake.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await asyncio.sleep(self.delay)
                self._wake.clear()
                version, text = self.version, self.text
                self._task = asyncio.create_task(self.generate(text))
                result = await self._task
                if version == self.version:
                    self._result = (text, result)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
            except Exception:
                # Speculation is optional. Final generation surfaces real errors.
                pass
            finally:
                self._task = None

    async def finish(self, final_text: str):
        result = self._result
        await self.close()
        return result[1] if result is not None and result[0] == final_text else None

    async def close(self):
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        self._result = None

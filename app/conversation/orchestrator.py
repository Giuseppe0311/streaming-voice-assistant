"""Sequential half-duplex turns with replaceable speculative work."""
import asyncio
from contextlib import aclosing
import logging
from app.conversation.state import State
from app.conversation.transcripts import Drafts

logger = logging.getLogger(__name__)


class Conversation:
    def __init__(self, microphone, transcriber, llm, speech, playback,
                 draft_delay=0.6, speculative_enabled=True):
        self.microphone, self.transcriber = microphone, transcriber
        self.llm, self.speech, self.playback = llm, speech, playback
        self.draft_delay, self.speculative_enabled = draft_delay, speculative_enabled
        self.state = State.STOPPED
        self.last_error = None
        self.last_error_stage = None
        self._current_stage = None
        self.partial_transcript = ''
        self.final_transcript = ''
        self.last_response = ''
        self.turn_number = 0

    async def turn(self):
        drafts = Drafts(self.llm.draft, self.draft_delay)
        self.partial_transcript = ''
        def partial(text):
            self.partial_transcript = text
            # Terminal output is deliberately opt-in through debug logging.
            logger.debug('Transcript: %s', text)
            if self.speculative_enabled:
                drafts.update(text)
        try:
            self._current_stage = 'transcription_connect'
            await self.transcriber.open()
            self.state = State.LISTENING
            self._current_stage = 'microphone_start'
            await self.microphone.start()
            self._current_stage = 'transcription_capture'
            text = await self.transcriber.capture(
                self.microphone, partial, lambda: setattr(self, "state", State.FINALIZING)
            )
            self.state = State.FINALIZING
            self._current_stage = 'transcription_finalize'
            await self.microphone.stop()
            await self.transcriber.close()
            self.final_transcript = text
            self._current_stage = 'draft_finalize'
            draft = await drafts.finish(text)
            if not text.strip():
                self.state = State.LISTENING
                self._current_stage = None
                return
            self.state = State.THINKING
            self._current_stage = 'response'
            answer = []
            async def audio_chunks():
                async with aclosing(self.llm.respond(text, draft)) as segments:
                    async for segment in segments:
                        answer.append(segment)
                        async with aclosing(self.speech.stream(segment)) as chunks:
                            async for chunk in chunks:
                                self.state = State.SPEAKING
                                yield chunk
            async with aclosing(audio_chunks()) as chunks:
                await self.playback.play(chunks)
            self.last_response = ' '.join(answer)
            self.llm.remember(text, self.last_response)
            self.turn_number += 1
            self.state = State.LISTENING
            self._current_stage = None
        finally:
            try:
                await drafts.close()
            finally:
                try:
                    await self.microphone.stop()
                finally:
                    await self.transcriber.close()

    async def run(self):
        self.last_error = None
        self.last_error_stage = None
        try:
            while True:
                await self.turn()
        except asyncio.CancelledError:
            self.state = State.STOPPED
            self._current_stage = None
            raise
        except Exception as exc:
            self.state = State.ERROR
            self.last_error = type(exc).__name__
            self.last_error_stage = self._current_stage or 'unknown'
            logger.error('Voice stopped (%s at %s); check configuration and restart the process',
                         self.last_error, self.last_error_stage)
        finally:
            results = await asyncio.gather(
                self.microphone.stop(), self.transcriber.close(), self.playback.close(),
                return_exceptions=True,
            )
            if any(isinstance(result, BaseException) for result in results):
                self.state = State.ERROR
                self.last_error = "cleanup_failed"
                self.last_error_stage = "cleanup"
                logger.error("Voice resource cleanup failed; restart the process")

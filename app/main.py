"""Application composition. Importing this module never opens audio or API clients."""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import router
from app.config import Settings

logger = logging.getLogger(__name__)


def build_conversation(client, api_key, settings):
    from app.audio.microphone import Microphone
    from app.audio.playback import Playback
    from app.conversation.orchestrator import Conversation
    from app.providers.llm import LanguageModel
    from app.providers.speech import Speech
    from app.providers.transcription import Transcription
    from app.tools.builtins import default_registry
    from app.tools.executor import ToolExecutor

    executor = ToolExecutor(default_registry(), timeout=settings.tool_timeout)
    return Conversation(
        Microphone(settings.input_device), Transcription(api_key, settings),
        LanguageModel(client, settings, executor),
        Speech(client, settings.tts_model, settings.tts_voice), Playback(settings.output_device),
        draft_delay=settings.draft_delay, speculative_enabled=settings.speculative_enabled,
    )


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(application):
        if settings is None:
            load_dotenv(override=False)
            config = Settings.from_env()
        else:
            config = settings
        task = client = None
        try:
            if config.voice_enabled:
                key = os.getenv('OPENAI_API_KEY')
                if not key:
                    application.state.voice_state = 'error'
                    application.state.last_error = 'missing_api_key'
                    application.state.last_error_stage = 'startup'
                else:
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(api_key=key, timeout=config.request_timeout, max_retries=0)
                    voice = build_conversation(client, key, config)
                    application.state.conversation = voice
                    logger.info('Voice enabled. The assistant voice is AI-generated.')
                    task = asyncio.create_task(voice.run(), name='voice-conversation')
            yield
        finally:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if client:
                await client.close()

    application = FastAPI(title='Streaming Voice Assistant', lifespan=lifespan)
    application.state.conversation = None
    application.state.voice_state = 'disabled'
    application.state.last_error = None
    application.state.last_error_stage = None
    application.include_router(router)
    return application


app = create_app()

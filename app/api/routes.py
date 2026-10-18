from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get('/')
@router.get('/health')
async def health(request: Request):
    runtime = request.app.state
    voice = runtime.conversation
    state = voice.state.value if voice else runtime.voice_state
    error = voice.last_error if voice else runtime.last_error
    error_stage = voice.last_error_stage if voice else runtime.last_error_stage
    return JSONResponse({'status': 'error' if state == 'error' else 'ok',
                          'service': 'streaming-voice-assistant',
                          'voice_state': state, 'last_error': error,
                          'last_error_stage': error_stage},
                         status_code=503 if state == 'error' else 200)


@router.get('/conversation')
async def conversation(request: Request):
    """Local-only transcript inspection; never expose this unauthenticated API."""
    voice = request.app.state.conversation
    return {'state': voice.state.value if voice else request.app.state.voice_state,
            'turn': voice.turn_number if voice else 0,
             'partial_transcript': voice.partial_transcript if voice else '',
             'final_transcript': voice.final_transcript if voice else '',
             'last_response': voice.last_response if voice else '',
             'last_error': voice.last_error if voice else request.app.state.last_error,
             'last_error_stage': voice.last_error_stage if voice else request.app.state.last_error_stage}

"""HTTP boot must remain safe without credentials or audio devices."""
from fastapi.testclient import TestClient
import app.main as main
from app.config import Settings
from app.main import create_app


def test_health_when_voice_disabled():
    with TestClient(create_app(Settings())) as client:
        assert client.get('/').json()['voice_state'] == 'disabled'
        assert client.get('/health').status_code == 200
        assert client.get('/conversation').json()['partial_transcript'] == ''


def test_missing_key_only_fails_when_voice_is_enabled(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(create_app(Settings(voice_enabled=True))) as client:
        body = client.get('/health').json()
        assert body['voice_state'] == 'error'
        assert body['last_error'] == 'missing_api_key'
        assert body['last_error_stage'] == 'startup'
        assert client.get('/health').status_code == 503


def test_default_runtime_loads_dotenv_before_settings(monkeypatch):
    events = []

    def load_dotenv(*, override):
        events.append(('dotenv', override))

    def from_env():
        events.append(('settings', None))
        return Settings()

    monkeypatch.setattr(main, 'load_dotenv', load_dotenv)
    monkeypatch.setattr(main.Settings, 'from_env', from_env)

    with TestClient(main.create_app()) as client:
        assert client.get('/health').status_code == 200

    assert events == [('dotenv', False), ('settings', None)]

from enum import StrEnum


class State(StrEnum):
    DISABLED = 'disabled'
    LISTENING = 'listening'
    FINALIZING = 'finalizing'
    THINKING = 'thinking'
    SPEAKING = 'speaking'
    STOPPED = 'stopped'
    ERROR = 'error'

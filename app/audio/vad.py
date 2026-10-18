"""Small client-side energy VAD for PCM16 mono at 24 kHz."""
import math
import struct


class SilenceDetector:
    def __init__(self, threshold=0.015, silence_seconds=0.9,
                 min_speech_seconds=0.2, max_turn_seconds=60,
                 release_ratio=0.5):
        self.threshold = threshold
        self.release_ratio = release_ratio
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_turn_seconds = max_turn_seconds
        self.speech_seconds = self.quiet_seconds = self.elapsed = 0.0
        self.started = False
        self.discarded = False

    @property
    def confirmed(self) -> bool:
        return self.speech_seconds + 1e-9 >= self.min_speech_seconds

    def feed(self, pcm: bytes) -> bool:
        self.discarded = False
        if not pcm or len(pcm) % 2:
            raise ValueError('Expected complete PCM16 samples')
        samples = struct.unpack(f'<{len(pcm) // 2}h', pcm)
        duration = len(samples) / 24000
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
        # Once speech is confirmed, a lower release threshold keeps quiet
        # fillers and trailing syllables from being mistaken for silence.
        active_threshold = self.threshold * self.release_ratio if self.confirmed else self.threshold
        if rms >= active_threshold:
            self.started = True
            self.speech_seconds += duration
            self.quiet_seconds = 0
        elif self.started:
            self.quiet_seconds += duration
        if self.started:
            self.elapsed += duration
        quiet_ended = self.quiet_seconds + 1e-9 >= self.silence_seconds
        limit_reached = self.elapsed + 1e-9 >= self.max_turn_seconds
        if self.started and not self.confirmed and (quiet_ended or limit_reached):
            # A click must not leave a candidate turn open for max_turn_seconds.
            self.started = False
            self.discarded = True
            self.speech_seconds = self.quiet_seconds = self.elapsed = 0.0
            return False
        return self.confirmed and (limit_reached or quiet_ended)

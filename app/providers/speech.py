from contextlib import aclosing


class Speech:
    def __init__(self, client, model: str, voice: str):
        self.client, self.model, self.voice = client, model, voice

    async def stream(self, text: str):
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model, voice=self.voice, input=text, response_format='pcm',
        ) as response:
            async with aclosing(response.iter_bytes(chunk_size=4800)) as chunks:
                async for chunk in chunks:
                    yield chunk

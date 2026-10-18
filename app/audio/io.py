"""Do not close audio streams while their blocking operations are in flight."""
import asyncio


async def blocking(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
            if not task.done():
                # A second cancellation must not cancel the wrapper while its
                # underlying thread still owns an audio stream operation.
                continue
            if not task.cancelled():
                task.exception()  # Retrieve any concurrent device failure.
            raise cancellation
        except Exception:
            if cancellation is not None:
                raise cancellation from None
            raise
        else:
            if cancellation is not None:
                raise cancellation
            return result

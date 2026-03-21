import edge_tts
from pathlib import Path
import time
import re
import asyncio


VOICE_CANDIDATES = [
    "en-IE-EmilyNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
]


def _normalize_text(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Edge can fail more often on very long utterances; cap safely.
    return cleaned[:1200] if len(cleaned) > 1200 else cleaned


async def voice(text, output_file=None):
    safe_text = _normalize_text(text)
    if not safe_text:
        safe_text = "Sorry, I had trouble speaking that response."

    audio_dir = Path(__file__).resolve().parent / "audio"
    audio_dir.mkdir(exist_ok=True)

    if output_file is None:
        output_file = audio_dir / f"voice_{int(time.time() * 1000)}.mp3"
    else:
        output_file = Path(output_file)

    last_error = None

    for voice_name in VOICE_CANDIDATES:
        for attempt in range(2):
            try:
                communicator = edge_tts.Communicate(safe_text, voice_name)
                await communicator.save(str(output_file))
                return output_file.name
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.2 * (attempt + 1))

    raise RuntimeError(f"Edge TTS failed after retries: {last_error}")
